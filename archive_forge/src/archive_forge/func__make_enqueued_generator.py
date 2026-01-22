import functools
import math
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _make_enqueued_generator(generator, workers=1, use_multiprocessing=False, max_queue_size=10, shuffle=False):
    """Create a buffered queue of next elements of the generator."""
    is_sequence = isinstance(generator, data_utils.Sequence)
    enqueuer = None
    if workers > 0:
        if is_sequence:
            enqueuer = data_utils.OrderedEnqueuer(generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
        else:
            enqueuer = data_utils.GeneratorEnqueuer(generator, use_multiprocessing=use_multiprocessing)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
    elif is_sequence:
        output_generator = data_utils.iter_sequence_infinite(generator)
    else:
        output_generator = generator
    return (output_generator, enqueuer)