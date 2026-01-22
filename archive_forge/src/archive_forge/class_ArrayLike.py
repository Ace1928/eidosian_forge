import collections
import numpy as np
import tensorflow.compat.v2 as tf
class ArrayLike:

    def __init__(self, values):
        self.values = values

    def __array__(self):
        return np.array(self.values)