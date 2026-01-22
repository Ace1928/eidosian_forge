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
def _make_train_function(self):
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    self._check_trainable_weights_consistency()
    if isinstance(self.optimizer, list):
        raise ValueError('The `optimizer` in `compile` should be a single optimizer.')
    if getattr(self, 'train_function', None) is None or has_recompiled:
        current_trainable_state = self._get_trainable_state()
        self._set_trainable_state(self._compiled_trainable_state)
        inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
        if not isinstance(backend.symbolic_learning_phase(), int):
            inputs += [backend.symbolic_learning_phase()]
        with backend.get_graph().as_default():
            with backend.name_scope('training'):
                updates = self.optimizer.get_updates(params=self._collected_trainable_weights, loss=self.total_loss)
                updates += self.get_updates_for(None)
                updates += self.get_updates_for(self.inputs)
            metrics = self._get_training_eval_metrics()
            metrics_tensors = [m._call_result for m in metrics if hasattr(m, '_call_result')]
        with backend.name_scope('training'):
            fn = backend.function(inputs, [self.total_loss] + metrics_tensors, updates=updates, name='train_function', **self._function_kwargs)
            setattr(self, 'train_function', fn)
        self._set_trainable_state(current_trainable_state)