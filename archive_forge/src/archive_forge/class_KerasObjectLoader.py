import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
class KerasObjectLoader(object):
    """Loader that recreates Keras objects (e.g. layers, models).

  Layers and models are revived from either the config or SavedModel following
  these rules:
  1. If object is a graph network (i.e. Sequential or Functional) then it will
     be initialized using the structure from the config only after the children
     layers have been created. Graph networks must be initialized with inputs
     and outputs, so all child layers must be created beforehand.
  2. If object's config exists and the class can be found, then revive from
     config.
  3. Object may have already been created if its parent was revived from config.
     In this case, do nothing.
  4. If nothing of the above applies, compose the various artifacts from the
     SavedModel to create a subclassed layer or model. At this time, custom
     metrics are not supported.

  """

    def __init__(self, metadata, object_graph_def):
        self._metadata = {x.node_id: x for x in metadata.nodes}
        self._proto = object_graph_def
        self._node_paths = {node_data.node_id: node_data.node_path for node_data in metadata.nodes}
        self.loaded_nodes = {}
        self._traversed_nodes_from_config = set()
        self.model_layer_dependencies = {}
        self._models_to_reconstruct = []

    def del_tracking(self):
        """Removes tracked references that are only used when loading the model."""
        for node in self.loaded_nodes.values():
            node = node[0]
            if not isinstance(node, base_layer.Layer):
                continue
            for name in PUBLIC_ATTRIBUTES:
                node._delete_tracking(name)
            if isinstance(node, functional_lib.Functional):
                dependencies = list(node._self_unconditional_dependency_names)
                for name in dependencies:
                    if re.match('^layer(_with_weights)?-[\\d+]', name) is not None:
                        node._delete_tracking(name)

    def _add_children_recreated_from_config(self, obj, proto, node_id):
        """Recursively records objects recreated from config."""
        if node_id in self._traversed_nodes_from_config:
            return
        parent_path = self._node_paths[node_id]
        self._traversed_nodes_from_config.add(node_id)
        obj._maybe_initialize_trackable()
        if isinstance(obj, base_layer.Layer) and (not obj.built):
            metadata = json_utils.decode(self._metadata[node_id].metadata)
            self._try_build_layer(obj, node_id, metadata.get('build_input_shape'))
        children = []
        for reference in proto.children:
            obj_child = obj._lookup_dependency(reference.local_name)
            children.append((obj_child, reference.node_id, reference.local_name))
        metric_list_node_id = self._search_for_child_node(node_id, [constants.KERAS_ATTR, 'layer_metrics'])
        if metric_list_node_id is not None and hasattr(obj, '_metrics'):
            obj_metrics = {m.name: m for m in obj._metrics}
            for reference in self._proto.nodes[metric_list_node_id].children:
                metric = obj_metrics.get(reference.local_name)
                if metric is not None:
                    metric_path = '{}.layer_metrics.{}'.format(constants.KERAS_ATTR, reference.local_name)
                    children.append((metric, reference.node_id, metric_path))
        for obj_child, child_id, child_name in children:
            child_proto = self._proto.nodes[child_id]
            if not isinstance(obj_child, trackable.Trackable):
                continue
            if child_proto.user_object.identifier in revived_types.registered_identifiers():
                setter = revived_types.get_setter(child_proto.user_object)
            elif obj_child._object_identifier in constants.KERAS_OBJECT_IDENTIFIERS:
                setter = _revive_setter
            else:
                setter = setattr
            if child_id in self.loaded_nodes:
                if self.loaded_nodes[child_id][0] is not obj_child:
                    logging.warning('Looks like there is an object (perhaps variable or layer) that is shared between different layers/models. This may cause issues when restoring the variable values. Object: {}'.format(obj_child))
                continue
            if child_proto.WhichOneof('kind') == 'variable' and child_proto.variable.name:
                obj_child._handle_name = child_proto.variable.name + ':0'
            if isinstance(obj_child, data_structures.TrackableDataStructure):
                setter = lambda *args: None
            child_path = '{}.{}'.format(parent_path, child_name)
            self._node_paths[child_id] = child_path
            self._add_children_recreated_from_config(obj_child, child_proto, child_id)
            self.loaded_nodes[child_id] = (obj_child, setter)

    def load_layers(self, compile=True):
        """Load all layer nodes from the metadata."""
        metric_list = []
        for node_metadata in self._metadata.values():
            if node_metadata.identifier == constants.METRIC_IDENTIFIER:
                metric_list.append(node_metadata)
                continue
            self.loaded_nodes[node_metadata.node_id] = self._load_layer(node_metadata.node_id, node_metadata.identifier, node_metadata.metadata)
        for node_metadata in metric_list:
            try:
                self.loaded_nodes[node_metadata.node_id] = self._load_layer(node_metadata.node_id, node_metadata.identifier, node_metadata.metadata)
            except ValueError:
                if compile:
                    raise
                logging.warning('Unable to restore custom metric. Please ensure that the layer implements `get_config` and `from_config` when saving. In addition, please use the `custom_objects` arg when calling `load_model()`.')

    def _load_layer(self, node_id, identifier, metadata):
        """Load a single layer from a SavedUserObject proto."""
        metadata = json_utils.decode(metadata)
        if node_id in self.loaded_nodes:
            node, setter = self.loaded_nodes[node_id]
            _maybe_add_serialized_attributes(node, metadata)
            config = metadata.get('config')
            if _is_graph_network(node) and generic_utils.validate_config(config):
                child_nodes = self._get_child_layer_node_ids(node_id)
                self.model_layer_dependencies[node_id] = (node, child_nodes)
                if not child_nodes:
                    self._models_to_reconstruct.append(node_id)
            return (node, setter)
        obj, setter = self._revive_from_config(identifier, metadata, node_id)
        if obj is None:
            obj, setter = revive_custom_object(identifier, metadata)
        _maybe_add_serialized_attributes(obj, metadata)
        return (obj, setter)

    def _revive_from_config(self, identifier, metadata, node_id):
        """Revives a layer/model from config, or returns None."""
        if identifier == constants.METRIC_IDENTIFIER:
            obj = self._revive_metric_from_config(metadata)
        else:
            obj = self._revive_graph_network(identifier, metadata, node_id) or self._revive_layer_or_model_from_config(metadata, node_id)
        if obj is None:
            return (None, None)
        setter = self._config_node_setter(_revive_setter)
        self._add_children_recreated_from_config(obj, self._proto.nodes[node_id], node_id)
        return (obj, setter)

    def _revive_graph_network(self, identifier, metadata, node_id):
        """Revives a graph network from config."""
        config = metadata.get('config')
        if not generic_utils.validate_config(config):
            return None
        class_name = compat.as_str(metadata['class_name'])
        if generic_utils.get_registered_object(class_name) is not None:
            return None
        model_is_functional_or_sequential = metadata.get('is_graph_network', False) or class_name == 'Sequential' or class_name == 'Functional'
        if not model_is_functional_or_sequential:
            return None
        if class_name == 'Sequential':
            model = models_lib.Sequential(name=config['name'])
        elif identifier == constants.SEQUENTIAL_IDENTIFIER:
            model = models_lib.Sequential(name=class_name)
        else:
            model = models_lib.Functional(inputs=[], outputs=[], name=config['name'])
        layers = self._get_child_layer_node_ids(node_id)
        self.model_layer_dependencies[node_id] = (model, layers)
        if not layers:
            self._models_to_reconstruct.append(node_id)
        return model

    def _revive_layer_or_model_from_config(self, metadata, node_id):
        """Revives a layer/custom model from config; returns None if infeasible."""
        class_name = metadata.get('class_name')
        config = metadata.get('config')
        shared_object_id = metadata.get('shared_object_id')
        must_restore_from_config = metadata.get('must_restore_from_config')
        if not generic_utils.validate_config(config):
            return None
        try:
            obj = layers_module.deserialize(generic_utils.serialize_keras_class_and_config(class_name, config, shared_object_id=shared_object_id))
        except ValueError:
            if must_restore_from_config:
                raise RuntimeError('Unable to restore a layer of class {cls}. Layers of class {cls} require that the class be provided to the model loading code, either by registering the class using @keras.utils.register_keras_serializable on the class def and including that file in your program, or by passing the class in a keras.utils.CustomObjectScope that wraps this load call.'.format(cls=class_name))
            else:
                return None
        obj._name = metadata['name']
        if metadata.get('trainable') is not None:
            obj.trainable = metadata['trainable']
        if metadata.get('dtype') is not None:
            obj._set_dtype_policy(metadata['dtype'])
        if metadata.get('stateful') is not None:
            obj.stateful = metadata['stateful']
        if isinstance(obj, training_lib.Model):
            save_spec = metadata.get('save_spec')
            if save_spec is not None:
                obj._set_save_spec(save_spec)
        build_input_shape = metadata.get('build_input_shape')
        built = self._try_build_layer(obj, node_id, build_input_shape)
        if not built:
            return None
        return obj

    def _revive_metric_from_config(self, metadata):
        """Revives a metric object using the config saved in the metadata."""
        class_name = compat.as_str(metadata['class_name'])
        config = metadata.get('config')
        if not generic_utils.validate_config(config):
            return None
        try:
            obj = metrics.deserialize(generic_utils.serialize_keras_class_and_config(class_name, config))
        except ValueError:
            return None
        build_input_shape = metadata.get('build_input_shape')
        if build_input_shape is not None and hasattr(obj, '_build'):
            obj._build(build_input_shape)
        return obj

    def _try_build_layer(self, obj, node_id, build_input_shape):
        """Attempts to build the layer."""
        if obj.built or hasattr(obj.build, '_is_default'):
            obj.built = True
            return True
        if build_input_shape is None:
            build_input_shape = self._infer_inputs(node_id, convert_to_shapes=True)
        if build_input_shape is not None:
            obj.build(build_input_shape)
            base_layer.Layer.build(obj, build_input_shape)
            return True
        return False

    def _load_edges(self):
        """Add edges for all nodes that are not waiting on initialization."""
        for node_id, proto in enumerate(self._proto.nodes):
            if node_id not in self.model_layer_dependencies:
                self._add_object_graph_edges(proto, node_id)

    def get_path(self, node_id):
        return self._node_paths[node_id]

    def finalize_objects(self):
        """Finish setting up Keras objects.

    This function is executed after all objects and functions have been created.
    Call functions and losses are attached to each layer, and once all layers
    have been fully set up, graph networks are initialized.

    Subclassed models that are revived from the SavedModel are treated like
    layers, and have their call/loss functions attached here.
    """
        layers_revived_from_config = []
        layers_revived_from_saved_model = []
        for node_id, (node, _) in self.loaded_nodes.items():
            if not isinstance(node, base_layer.Layer) or node_id in self.model_layer_dependencies:
                continue
            self._unblock_model_reconstruction(node_id, node)
            if isinstance(node, input_layer.InputLayer):
                continue
            elif isinstance(node, metrics.Metric):
                continue
            if isinstance(node, (RevivedLayer, RevivedInputLayer)):
                layers_revived_from_saved_model.append(node)
            else:
                layers_revived_from_config.append(node)
        _finalize_saved_model_layers(layers_revived_from_saved_model)
        _finalize_config_layers(layers_revived_from_config)
        self._reconstruct_all_models()

    def _unblock_model_reconstruction(self, layer_id, layer):
        """Removes layer from blocking model reconstruction."""
        for model_id, v in self.model_layer_dependencies.items():
            _, layers = v
            if layer_id not in layers:
                continue
            layers[layers.index(layer_id)] = layer
            if all((isinstance(x, base_layer.Layer) for x in layers)):
                self._models_to_reconstruct.append(model_id)

    def _reconstruct_all_models(self):
        """Reconstructs the network structure of all models."""
        all_initialized_models = set()
        while self._models_to_reconstruct:
            model_id = self._models_to_reconstruct.pop(0)
            all_initialized_models.add(model_id)
            model, layers = self.model_layer_dependencies[model_id]
            self._reconstruct_model(model_id, model, layers)
            _finalize_config_layers([model])
        if all_initialized_models != set(self.model_layer_dependencies.keys()):
            uninitialized_model_ids = set(self.model_layer_dependencies.keys()) - all_initialized_models
            uninitialized_model_names = [self.model_layer_dependencies[model_id][0].name for model_id in uninitialized_model_ids]
            raise ValueError('Error when loading from SavedModel -- the following models could not be initialized: {}'.format(uninitialized_model_names))

    def _reconstruct_model(self, model_id, model, layers):
        """Reconstructs the network structure."""
        config = json_utils.decode(self._metadata[model_id].metadata)['config']
        if model.inputs:
            pass
        elif isinstance(model, models_lib.Sequential):
            if not layers or not isinstance(layers[0], input_layer.InputLayer):
                if config['layers'][0]['class_name'] == 'InputLayer':
                    layers.insert(0, input_layer.InputLayer.from_config(config['layers'][0]['config']))
                elif 'batch_input_shape' in config['layers'][0]['config']:
                    batch_input_shape = config['layers'][0]['config']['batch_input_shape']
                    layers.insert(0, input_layer.InputLayer(input_shape=batch_input_shape[1:], batch_size=batch_input_shape[0], dtype=layers[0].dtype, name=layers[0].name + '_input'))
            model.__init__(layers, name=config['name'])
            if not model.inputs:
                first_layer = self._get_child_layer_node_ids(model_id)[0]
                input_specs = self._infer_inputs(first_layer)
                input_shapes = self._infer_inputs(first_layer, convert_to_shapes=True)
                model._set_inputs(input_specs)
                if not model.built and (not isinstance(input_specs, dict)):
                    model.build(input_shapes)
        else:
            inputs, outputs, created_layers = functional_lib.reconstruct_from_config(config, created_layers={layer.name: layer for layer in layers})
            model.__init__(inputs, outputs, name=config['name'])
            functional_lib.connect_ancillary_layers(model, created_layers)
        _set_network_attributes_from_metadata(model)
        self._unblock_model_reconstruction(model_id, model)

    def _get_child_layer_node_ids(self, node_id):
        """Returns the node ids of each layer in a Sequential/Functional model."""
        num_layers = 0
        child_layers = {}
        pattern = re.compile('layer-(\\d+)')
        for child in self._proto.nodes[node_id].children:
            m = pattern.match(child.local_name)
            if m is None:
                continue
            layer_n = int(m.group(1))
            num_layers = max(layer_n + 1, num_layers)
            child_layers[layer_n] = child.node_id
        ordered = []
        for n in range(num_layers):
            child = child_layers.get(n)
            if child is None:
                break
            ordered.append(child)
        return ordered

    def _search_for_child_node(self, parent_id, path_to_child):
        """Returns node id of child node.

    A helper method for traversing the object graph proto.

    As an example, say that the object graph proto in the SavedModel contains an
    object with the following child and grandchild attributes:

    `parent.child_a.child_b`

    This method can be used to retrieve the node id of `child_b` using the
    parent's node id by calling:

    `_search_for_child_node(parent_id, ['child_a', 'child_b'])`.

    Args:
      parent_id: node id of parent node
      path_to_child: list of children names.

    Returns:
      node_id of child, or None if child isn't found.
    """
        if not path_to_child:
            return parent_id
        for child in self._proto.nodes[parent_id].children:
            if child.local_name == path_to_child[0]:
                return self._search_for_child_node(child.node_id, path_to_child[1:])
        return None

    def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
        """Infers input shape of layer from SavedModel functions."""
        call_fn_id = self._search_for_child_node(layer_node_id, ['call_and_return_all_conditional_losses'])
        if call_fn_id is None:
            return None
        concrete_functions = self._proto.nodes[call_fn_id].function.concrete_functions
        if not concrete_functions:
            return None
        call_fn_name = concrete_functions[0]
        call_fn_proto = self._proto.concrete_functions[call_fn_name]
        structured_input_signature = nested_structure_coder.decode_proto(call_fn_proto.canonicalized_input_signature)
        inputs = structured_input_signature[0][0]
        if convert_to_shapes:
            return nest.map_structure(lambda spec: spec.shape, inputs)
        else:
            return inputs

    def _config_node_setter(self, setter):
        """Creates edges for nodes that are recreated from config."""

        def setattr_wrapper(obj, name, value):
            if obj._lookup_dependency(name) is None:
                setter(obj, name, value)
        return setattr_wrapper