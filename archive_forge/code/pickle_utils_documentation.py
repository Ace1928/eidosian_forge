import os
import tempfile
import tensorflow.compat.v2 as tf
from keras.src.saving import saving_lib
Convert a Keras Model into a bytecode representation for pickling.

    Args:
        model: Keras Model instance.

    Returns:
        Tuple that can be read by `deserialize_from_bytecode`.
    