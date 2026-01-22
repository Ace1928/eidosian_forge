import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class StopGradient(Operation):

    def __init__(self):
        super().__init__()

    def call(self, variable):
        return backend.core.stop_gradient(variable)

    def compute_output_spec(self, variable):
        return KerasTensor(variable.shape, dtype=variable.dtype)