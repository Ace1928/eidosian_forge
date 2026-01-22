import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def _update_step_fn(self, grads, trainable_variables):
    steps = self.gradient_accumulation_steps
    grads = [(grads[i] + self._accumulated_gradients[i]) / steps for i in range(len(grads))]
    self._backend_update_step(grads, trainable_variables, self.learning_rate)
    self._backend_reset_gradient_accumulators()