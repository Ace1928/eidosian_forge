import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator

    Assume an angle ø, then rotation matrix is defined by
    | cos(ø)   -sin(ø)  x_offset |
    | sin(ø)    cos(ø)  y_offset |
    |   0         0         1    |

    This function is returning the 8 elements barring the final 1 as a 1D array
    