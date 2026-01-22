import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _distributed_apply_gradients_fn(self, distribution, grads_and_vars, **kwargs):
    """`apply_gradients` using a `DistributionStrategy`."""

    def apply_grad_to_update_var(var, grad):
        if self.jit_compile:
            return self._update_step_xla(grad, var, id(self._var_key(var)))
        else:
            return self._update_step(grad, var)
    for grad, var in grads_and_vars:
        distribution.extended.update(var, apply_grad_to_update_var, args=(grad,), group=False)
    if self.use_ema:
        _, var_list = zip(*grads_and_vars)
        self._update_model_variables_moving_average(var_list)
        if self.ema_overwrite_frequency:
            should_overwrite_model_vars = (self.iterations + 1) % self.ema_overwrite_frequency == 0
            tf.cond(tf.cast(should_overwrite_model_vars, tf.bool), true_fn=lambda: self._overwrite_model_variables_with_average_value(var_list), false_fn=lambda: None)
    return self.iterations.assign_add(1)