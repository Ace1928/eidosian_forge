import collections
import math
import string
import numpy as np
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.activations.softmax import Softmax
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.layer import Layer
from keras.src.layers.regularization.dropout import Dropout
def _index_to_einsum_variable(i):
    """Coverts an index to a einsum variable name.

    We simply map indices to lowercase characters, e.g. 0 -> 'a', 1 -> 'b'.
    """
    return string.ascii_lowercase[i]