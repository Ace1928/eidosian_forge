from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _clone_sequential_model(model, input_tensors=None, layer_fn=_clone_layer):
    """Clone a `Sequential` model instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Args:
      model: Instance of `Sequential`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.
      layer_fn: callable to be applied on non-input layers in the model. By
          default it clones the layer. Another example is to preserve the layer
          to share the weights. This is required when we create a per-replica
          copy of the model with distribution strategy; we want the weights to
          be shared but still feed inputs separately so we create new input
          layers.

  Returns:
      An instance of `Sequential` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value or `layer_fn`
      argument value.
  """
    if not isinstance(model, Sequential):
        raise ValueError('Expected `model` argument to be a `Sequential` model instance, but got:', model)
    if not callable(layer_fn):
        raise ValueError('Expected `layer_fn` argument to be a callable.')
    layers = []
    layer_map = {}
    for layer in model._flatten_layers(include_self=False, recursive=False):
        if isinstance(layer, InputLayer) and input_tensors is not None:
            continue
        cloned_layer = _clone_layer(layer) if isinstance(layer, InputLayer) else layer_fn(layer)
        layers.append(cloned_layer)
        layer_map[layer] = cloned_layer
    layers, ancillary_layers = _remove_ancillary_layers(model, layer_map, layers)
    if input_tensors is None:
        cloned_model = Sequential(layers=layers, name=model.name)
    elif len(generic_utils.to_list(input_tensors)) != 1:
        raise ValueError('To clone a `Sequential` model, we expect  at most one tensor as part of `input_tensors`.')
    else:
        if isinstance(input_tensors, tuple):
            input_tensors = list(input_tensors)
        x = generic_utils.to_list(input_tensors)[0]
        if backend.is_keras_tensor(x):
            origin_layer = x._keras_history.layer
            if isinstance(origin_layer, InputLayer):
                cloned_model = Sequential(layers=[origin_layer] + layers, name=model.name)
            else:
                raise ValueError('Cannot clone a `Sequential` model on top of a tensor that comes from a Keras layer other than an `InputLayer`. Use the functional API instead.')
        else:
            input_tensor = Input(tensor=x, name='input_wrapper_for_' + str(x.name))
            input_layer = input_tensor._keras_history.layer
            cloned_model = Sequential(layers=[input_layer] + layers, name=model.name)
    if not ancillary_layers:
        return cloned_model
    tensor_map = {}
    for depth, cloned_nodes in cloned_model._nodes_by_depth.items():
        nodes = model._nodes_by_depth[depth]
        for cloned_node, node in zip(cloned_nodes, nodes):
            if isinstance(cloned_node.output_tensors, list):
                for j, output_tensor in enumerate(cloned_node.output_tensors):
                    tensor_map[node.output_tensors[j]] = output_tensor
            else:
                tensor_map[node.output_tensors] = cloned_node.output_tensors
    new_nodes = _make_new_nodes({depth: nodes for depth, nodes in model._nodes_by_depth.items() if depth < 0}, layer_fn, layer_map, tensor_map)
    _insert_ancillary_layers(cloned_model, ancillary_layers, model.metrics_names, new_nodes)
    return cloned_model