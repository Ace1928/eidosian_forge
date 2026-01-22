import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_training_utils
from keras.src.engine import training_utils
from keras.src.utils import data_utils
from keras.src.utils import dataset_creator
from keras.src.utils import tf_utils
from tensorflow.python.distribute.input_lib import (
from tensorflow.python.eager import context
from tensorflow.python.framework import type_spec
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.data.ops import (
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.data.ops import range_op
from tensorflow.python.data.ops import from_tensors_op
from tensorflow.python.data.ops import from_tensor_slices_op
def _get_tensor_spec(t):
    return type_spec.type_spec_from_value(t)._with_tensor_ranks_only()