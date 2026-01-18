import threading
from tensorflow.python import tf2
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import convolutional_recurrent
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers import pooling
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import rnn_cell_wrapper_v2
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect as inspect
Instantiates a layer from a config dictionary.

  Args:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

  Returns:
      Layer instance (may be Model, Sequential, Network, Layer...)
  