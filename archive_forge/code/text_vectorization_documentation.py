import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.layers.preprocessing import string_lookup
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Sets vocabulary (and optionally document frequency) for this layer.

        This method sets the vocabulary and idf weights for this layer directly,
        instead of analyzing a dataset through 'adapt'. It should be used
        whenever the vocab (and optionally document frequency) information is
        already known.  If vocabulary data is already present in the layer, this
        method will replace it.

        Args:
          vocabulary: Either an array or a string path to a text file. If
            passing an array, can pass a tuple, list, 1D numpy array, or 1D
            tensor containing the vocbulary terms. If passing a file path, the
            file should contain one line per term in the vocabulary.
          idf_weights: A tuple, list, 1D numpy array, or 1D tensor of inverse
            document frequency weights with equal length to vocabulary. Must be
            set if `output_mode` is `"tf_idf"`. Should not be set otherwise.

        Raises:
          ValueError: If there are too many inputs, the inputs do not match, or
            input data is missing.
          RuntimeError: If the vocabulary cannot be set when this function is
            called. This happens when `"multi_hot"`, `"count"`, and "tf_idf"
            modes, if `pad_to_max_tokens` is False and the layer itself has
            already been called.
        