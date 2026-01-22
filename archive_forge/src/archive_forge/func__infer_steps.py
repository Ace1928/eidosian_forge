import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _infer_steps(self, steps, dataset):
    """Infers steps_per_epoch needed to loop through a dataset."""
    if steps == -1:
        self._log_indefinite_training_warning()
        return None
    if steps is not None:
        return steps
    adapter_steps = self._adapter.get_size()
    if adapter_steps is not None:
        return adapter_steps
    size = cardinality.cardinality(dataset)
    if size == cardinality.INFINITE and steps is None:
        raise ValueError('When passing an infinitely repeating dataset, please specify a `steps_per_epoch` value so that epoch level callbacks continue to work. The value can be arbitrary, or a number that you think correctly defines the size of an epoch. Epoch-level callbacks will then be called at this interval.')
    if size >= 0:
        return size.numpy().item()
    return None