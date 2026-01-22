import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def _get_type_spec(dataset):
    """Get the type spec of the dataset."""
    if isinstance(dataset, tuple):
        return tuple
    elif isinstance(dataset, list):
        return list
    elif isinstance(dataset, np.ndarray):
        return np.ndarray
    elif isinstance(dataset, dict):
        return dict
    elif isinstance(dataset, tf.data.Dataset):
        return tf.data.Dataset
    else:
        return None