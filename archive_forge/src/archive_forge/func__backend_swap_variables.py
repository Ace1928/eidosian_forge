from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
def _backend_swap_variables(self, optimizer):
    for var, average_var in zip(self.model.trainable_variables, optimizer._model_variables_moving_average):
        temporary_variable = ops.convert_to_numpy(var)
        var.assign(average_var)
        average_var.assign(temporary_variable)