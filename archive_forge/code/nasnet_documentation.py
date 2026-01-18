import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).

    Args:
      ip: Input tensor `x`
      p: Input tensor `p`
      filters: Number of output filters
      block_id: String block_id

    Returns:
      A Keras tensor
    