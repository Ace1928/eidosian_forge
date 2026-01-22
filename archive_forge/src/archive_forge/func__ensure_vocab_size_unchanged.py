import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
def _ensure_vocab_size_unchanged(self):
    if self.output_mode == INT or self.pad_to_max_tokens:
        return
    with tf.init_scope():
        new_vocab_size = self.vocabulary_size()
    if self._frozen_vocab_size is not None and new_vocab_size != self._frozen_vocab_size:
        raise RuntimeError(f'When using `output_mode={self.output_mode}` and `pad_to_max_tokens=False`, the vocabulary size cannot be changed after the layer is called. Old vocab size is {self._frozen_vocab_size}, new vocab size is {new_vocab_size}')