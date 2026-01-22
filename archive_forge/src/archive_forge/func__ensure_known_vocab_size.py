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
def _ensure_known_vocab_size(self):
    if self.output_mode == INT or self.pad_to_max_tokens:
        return
    if self._frozen_vocab_size is None:
        raise RuntimeError(f"When using `output_mode={self.output_mode}` and `pad_to_max_tokens=False`, you must set the layer's vocabulary before calling it. Either pass a `vocabulary` argument to the layer, or call `adapt` with some sample data.".format(self.output_mode))