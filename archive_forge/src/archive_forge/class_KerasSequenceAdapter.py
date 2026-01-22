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
class KerasSequenceAdapter(GeneratorDataAdapter):
    """Adapter that handles `keras.utils.Sequence`."""

    @staticmethod
    def can_handle(x, y=None):
        return isinstance(x, data_utils.Sequence)

    def __init__(self, x, y=None, sample_weights=None, shuffle=False, workers=1, use_multiprocessing=False, max_queue_size=10, model=None, **kwargs):
        if not is_none_or_empty(y):
            raise ValueError('`y` argument is not supported when using `keras.utils.Sequence` as input.')
        if not is_none_or_empty(sample_weights):
            raise ValueError('`sample_weight` argument is not supported when using `keras.utils.Sequence` as input.')
        self._size = len(x)
        self._shuffle_sequence = shuffle
        self._keras_sequence = x
        self._enqueuer = None
        super(KerasSequenceAdapter, self).__init__(x, shuffle=False, workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size, model=model, **kwargs)

    @staticmethod
    def _peek_and_restore(x):
        return (x[0], x)

    def _handle_multiprocessing(self, x, workers, use_multiprocessing, max_queue_size):
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                self._enqueuer = data_utils.OrderedEnqueuer(x, use_multiprocessing=use_multiprocessing, shuffle=self._shuffle_sequence)
                self._enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                return self._enqueuer.get()
        else:

            def generator_fn():
                order = range(len(x))
                if self._shuffle_sequence:
                    order = list(order)
                    random.shuffle(order)
                for i in order:
                    yield x[i]
        return generator_fn

    def get_size(self):
        return self._size

    def should_recreate_iterator(self):
        return True

    def on_epoch_end(self):
        if self._enqueuer:
            self._enqueuer.stop()
        self._keras_sequence.on_epoch_end()