import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
@trackable.no_automatic_dependency_tracking
def _set_input_attrs(self, inputs):
    """Sets attributes related to the inputs of the Model."""
    if self.inputs:
        raise ValueError('Model inputs are already set.')
    if self.__class__.__name__ == 'Sequential' and (not self.built):
        if tensor_util.is_tf_type(inputs):
            input_shape = (None,) + tuple(inputs.shape.as_list()[1:])
        elif isinstance(inputs, tensor_shape.TensorShape):
            input_shape = (None,) + tuple(inputs.as_list()[1:])
        elif isinstance(inputs, dict):
            if not training_utils_v1.is_feature_layer(self.layers[0]):
                raise ValueError("Passing a dictionary input to a Sequential Model which doesn't have FeatureLayer as the first layer is an error.")
            input_shape = (None,)
        else:
            input_shape = (None,) + tuple(inputs.shape[1:])
        self._build_input_shape = input_shape
    inputs = self._maybe_cast_inputs(inputs)
    model_inputs = training_utils_v1.ModelInputs(inputs)
    inputs = model_inputs.get_symbolic_inputs()
    self.inputs = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.input_names = model_inputs.get_input_names()
    self._feed_inputs = []
    self._feed_input_names = []
    self._feed_input_shapes = []
    for k, v in model_inputs.as_dict():
        if backend.is_placeholder(v):
            self._feed_input_names.append(k)
            self._feed_inputs.append(v)
            self._feed_input_shapes.append(backend.int_shape(v))
    return inputs