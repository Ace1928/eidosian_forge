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
def _process_target_tensor_for_compile(self, target_tensors):
    if self.run_eagerly:
        return [None for _ in self.output_names]
    if target_tensors is not None and (not (isinstance(target_tensors, list) and target_tensors == [])):
        if isinstance(target_tensors, list):
            if len(target_tensors) != len(self.outputs):
                raise ValueError('When passing a list as `target_tensors`, it should have one entry per model output. The model has %s outputs, but you passed target_tensors=%s' % (len(self.outputs), target_tensors))
        elif isinstance(target_tensors, dict):
            unexpected_target_tensor_names = set(target_tensors.keys()).difference(self.output_names)
            if unexpected_target_tensor_names:
                raise ValueError('Unknown entry in `target_tensors` dictionary: "{name}". Only expected the following keys: {keys}'.format(name=unexpected_target_tensor_names, keys=str(self.output_names)))
            tmp_target_tensors = []
            for name in self.output_names:
                tmp_target_tensors.append(target_tensors.get(name, None))
            target_tensors = tmp_target_tensors
        elif tensor_util.is_tf_type(target_tensors):
            target_tensors = [target_tensors]
        else:
            raise TypeError('Expected `target_tensors` to be a list or tuple or dict or a single tensor, but got:', target_tensors)
    else:
        target_tensors = [None for _ in self.output_names]
    return target_tensors