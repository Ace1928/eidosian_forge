import collections
import itertools
from functools import partial
import jax
import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import ops
from keras.src import optimizers as optimizers_module
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def _record_training_state_sharding_spec(self):
    self._trainable_variable_shardings = [v.value.sharding for v in self.trainable_variables]
    self._non_trainable_variable_shardings = [v.value.sharding for v in self.non_trainable_variables]
    if hasattr(self, 'optimizer') and self.optimizer is not None:
        self._optimizer_variable_shardings = [v.value.sharding for v in self.optimizer.variables]
    else:
        self._optimizer_variable_shardings = []
    self._metrics_variable_shardings = [v.value.sharding for v in self.metrics_variables]