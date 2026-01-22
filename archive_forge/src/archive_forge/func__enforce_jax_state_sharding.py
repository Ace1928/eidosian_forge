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
def _enforce_jax_state_sharding(self, trainable_variables=None, non_trainable_variables=None, optimizer_variables=None, metrics_variables=None):
    """Enforce the sharding spec constraint for all the training state.

        Since the output of the train/eval step will be used as inputs to next
        step, we need to ensure that they have the same sharding spec, so that
        jax.jit won't have to recompile the train/eval function.

        Note that this function will also rely on the recorded sharding spec
        for each of states.

        This function is expected to be called within the jitted train/eval
        function, especially around the end of the function.
        """
    trainable_variables = trainable_variables or []
    non_trainable_variables = non_trainable_variables or []
    optimizer_variables = optimizer_variables or []
    metrics_variables = metrics_variables or []
    for i in range(len(trainable_variables)):
        trainable_variables[i] = jax.lax.with_sharding_constraint(trainable_variables[i], self._trainable_variable_shardings[i])
    for i in range(len(non_trainable_variables)):
        non_trainable_variables[i] = jax.lax.with_sharding_constraint(non_trainable_variables[i], self._non_trainable_variable_shardings[i])
    for i in range(len(optimizer_variables)):
        optimizer_variables[i] = jax.lax.with_sharding_constraint(optimizer_variables[i], self._optimizer_variable_shardings[i])
    for i in range(len(metrics_variables)):
        metrics_variables[i] = jax.lax.with_sharding_constraint(metrics_variables[i], self._metrics_variable_shardings[i])
    return (trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables)