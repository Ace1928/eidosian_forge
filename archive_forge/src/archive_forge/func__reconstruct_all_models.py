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