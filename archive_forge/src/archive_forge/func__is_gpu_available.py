import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _is_gpu_available():
    return bool(tf.config.list_logical_devices('GPU'))