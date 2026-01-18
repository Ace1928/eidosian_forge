import json
import numpy as np
from keras.src.preprocessing.sequence import _remove_long_seq
from keras.src.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
Returns labels as a list of strings with indices matching training data.

    Reference:

    - [Reuters Dataset](https://martin-thoma.com/nlp-reuters/)
    