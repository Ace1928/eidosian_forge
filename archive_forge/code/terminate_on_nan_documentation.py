import numpy as np
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils
Callback that terminates training when a NaN loss is encountered.