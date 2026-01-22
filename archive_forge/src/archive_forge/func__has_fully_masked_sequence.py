import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _has_fully_masked_sequence(mask):
    return tf.reduce_any(tf.reduce_all(tf.logical_not(tf.cast(mask, dtype='bool')), axis=1))