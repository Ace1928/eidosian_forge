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
def _inverse_document_frequency(self, token_document_counts, num_documents):
    """Computes the inverse-document-frequency (IDF) component of "tf_idf".

        Uses the default weighting scheme described in
        https://en.wikipedia.org/wiki/Tf%E2%80%93idf.

        Args:
          token_document_counts: An array of the # of documents each token
            appears in.
          num_documents: An int representing the total number of documents

        Returns:
          An array of "inverse document frequency" weights.
        """
    return tf.math.log(1 + num_documents / (1 + token_document_counts))