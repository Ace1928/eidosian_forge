from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
def _tf_finalize_ema_values(self, optimizer):
    for var, average_var in zip(self.model.trainable_variables, optimizer._model_variables_moving_average):
        if isinstance(var, backend.Variable):
            var = var.value
        if isinstance(average_var, backend.Variable):
            average_var = average_var.value
        optimizer._distribution_strategy.extended.update(average_var, lambda a, b: a.assign(b), args=(var,))