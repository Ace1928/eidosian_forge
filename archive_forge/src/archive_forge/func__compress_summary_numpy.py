import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _compress_summary_numpy(summary, epsilon):
    """Compress a summary with numpy."""
    if summary.shape[1] * epsilon < 1:
        return summary
    percents = epsilon + np.arange(0.0, 1.0, epsilon)
    cum_weights = summary[1].cumsum()
    cum_weight_percents = cum_weights / cum_weights[-1]
    new_bins = np.interp(percents, cum_weight_percents, summary[0])
    cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
    new_weights = cum_weights - np.concatenate((np.array([0]), cum_weights[:-1]))
    summary = np.stack((new_bins, new_weights))
    return summary.astype(np.float32)