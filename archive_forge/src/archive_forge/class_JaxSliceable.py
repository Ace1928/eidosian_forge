import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class JaxSliceable(Sliceable):

    def __getitem__(self, indices):
        return self.array[indices, ...]

    @classmethod
    def convert_to_numpy(cls, x):
        from keras.src.backend.jax.core import convert_to_numpy
        return convert_to_numpy(x)