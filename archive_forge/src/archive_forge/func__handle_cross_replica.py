from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _handle_cross_replica(distribution, grads, trainable_variables):
    finite_per_replica = distribution.extended.call_for_each_replica(self.check_finite, args=(grads,))
    finite = distribution.experimental_local_results(finite_per_replica)[0]

    def apply_fn():
        distribution.extended.call_for_each_replica(self._stateful_handle_finite_grads, args=(grads, trainable_variables))
    ops.cond(finite, apply_fn, self._stateful_handle_non_finite_grads)