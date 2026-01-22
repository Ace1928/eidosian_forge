import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
def _get_compare_result_tolerance(key):
    """Returns tolerance to compare results."""
    if tf.test.is_gpu_available() and key.startswith(('weights_1', 'weights_2', 'predict_result')):
        return relaxed_tolerance
    return default_tolerance