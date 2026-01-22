from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _stateless_handle_non_finite_grads(self, optimizer_variables, trainable_variables):
    mapping = list(zip(self.variables, optimizer_variables))
    with backend.StatelessScope(state_mapping=mapping) as scope:
        self.step_counter.assign(0)
        self.dynamic_scale.assign(self.dynamic_scale / 2.0)
    new_optimizer_variables = []
    for v in self.variables:
        new_optimizer_variables.append(scope.get_current_value(v))
    return (trainable_variables, new_optimizer_variables)