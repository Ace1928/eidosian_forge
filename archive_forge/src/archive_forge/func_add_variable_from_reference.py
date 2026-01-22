import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def add_variable_from_reference(self, reference_variable, name=None, initializer='zeros'):
    if isinstance(reference_variable, backend.Variable):
        colocate_var = reference_variable.value
    else:
        colocate_var = reference_variable
    with self._distribution_strategy.extended.colocate_vars_with(colocate_var):
        return super().add_variable_from_reference(reference_variable, name=name, initializer=initializer)