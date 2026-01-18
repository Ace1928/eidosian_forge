import copy
from tensorflow.python.eager import def_function
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
A concrete tf.function that wraps the model's call function.