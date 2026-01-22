import functools
import itertools
import operator
import tensorflow as tf
from keras.src.backend.tensorflow.core import convert_to_tensor
def _nearest_indices_and_weights(coordinate):
    coordinate = coordinate if coordinate.dtype.is_integer else tf.round(coordinate)
    index = tf.cast(coordinate, tf.int32)
    weight = tf.constant(1, coordinate.dtype)
    return [(index, weight)]