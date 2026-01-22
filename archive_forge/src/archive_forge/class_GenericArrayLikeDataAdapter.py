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
class GenericArrayLikeDataAdapter(TensorLikeDataAdapter):
    """Adapter that handles array-like data without forcing it into memory.

  This adapter handles array-like datasets that may be too big to fully
  fit into memory.

  Specifically, this adapter handles any Python class which implements:
  `__get_item__`, `__len__`, `shape`, and `dtype` with the same meanings
  as Numpy, but it ignores any case where all the inputs are Tensors or Numpy
  arrays (because that case is handled by the base TensorLikeDataAdapter).

  It ignores scipy sparse matrices and Composite Tensors because those are
  handled by the CompositeTensorDataAdapter.

  It also does not handle lists/tuples of scalars, because those are handled
  by the ListsOfScalarsDataAdapter.
  """

    @staticmethod
    def can_handle(x, y=None):
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)

        def _is_array_like(v):
            """Return True if v is a Tensor, array, or is array-like."""
            return hasattr(v, '__getitem__') and hasattr(v, 'shape') and hasattr(v, 'dtype') and hasattr(v, '__len__')
        if not TensorLikeDataAdapter.can_handle(x, y) and (not CompositeTensorDataAdapter.can_handle(x, y)):
            return all((_is_array_like(v) for v in flat_inputs))
        else:
            return False

    def __init__(self, *args, **kwargs):
        logging.warning('Keras is training/fitting/evaluating on array-like data. Keras may not be optimized for this format, so if your input data format is supported by TensorFlow I/O (https://github.com/tensorflow/io) we recommend using that to load a Dataset instead.')
        super(GenericArrayLikeDataAdapter, self).__init__(*args, **kwargs)

    def slice_inputs(self, indices_dataset, inputs):
        """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
        flat_inputs = nest.flatten(inputs)

        def dynamic_shape_like(t):
            shape = list(t.shape)
            shape[0] = None
            return tuple(shape)
        flat_dtypes = [inp.dtype for inp in flat_inputs]
        contiguous = True
        if self._shuffle and self._shuffle != 'batch':
            contiguous = False

        def grab_batch(indices):
            """Grab a batch of data from the inputs."""

            def py_method(ind):

                def slice_array(data):
                    return training_utils.slice_arrays(data, ind.numpy(), contiguous=contiguous)
                return [slice_array(inp) for inp in flat_inputs]
            flat_out = script_ops.eager_py_func(py_method, [indices], flat_dtypes)
            for v, original_inp in zip(flat_out, flat_inputs):
                v.set_shape(dynamic_shape_like(original_inp))
            return nest.pack_sequence_as(inputs, flat_out)
        dataset = indices_dataset.map(grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)
        return dataset