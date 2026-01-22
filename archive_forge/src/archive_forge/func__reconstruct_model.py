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