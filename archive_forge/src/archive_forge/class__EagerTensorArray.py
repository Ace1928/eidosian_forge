import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
class _EagerTensorArray:
    """Eager-compatible implementation of TensorArray."""

    def __init__(self, dtype, size=None, dynamic_size=None, clear_after_read=None, tensor_array_name=None, handle=None, flow=None, infer_shape=True, element_shape=None, colocate_with_first_write_call=True, name=None):
        """Constructs a TensorArray compatible with eager execution.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: unused.
      handle: unsupported.
      flow: unsupported.
      infer_shape: used for error checking, same semantics as TensorArray.
      element_shape: used for error checking, same semantics as TensorArray.
      colocate_with_first_write_call: unsupported.
      name: unsupported.

    Raises:
      ValueError: handle or flow are supplied, or if size is not supplied.
    """
        del (flow, tensor_array_name, name)
        if handle is not None:
            raise ValueError('TensorArray handles are not supported when eager execution is enabled.')
        if size is None:
            raise ValueError('Size must be declared for TensorArrays when eager execution is enabled.')
        self._handle = None
        self._flow = constant_op.constant(0, dtype=dtypes.int32)
        self._infer_shape = infer_shape
        self._element_shape = tensor_shape.as_shape(element_shape)
        self._colocate_with_first_write_call = colocate_with_first_write_call
        self._dtype = dtypes.as_dtype(dtype).base_dtype
        self._dynamic_size = dynamic_size or False
        self._clear_after_read = True if clear_after_read is None else clear_after_read
        self._previously_read_indices = []
        if isinstance(size, ops.EagerTensor):
            size = size.numpy()
        self._tensor_array = [None for _ in range(size)]

    @property
    def flow(self):
        """For compatibility; flows are not meaningful when eager is enabled."""
        return self._flow

    @property
    def dtype(self):
        return self._dtype

    @property
    def handle(self):
        """For compatibility; handles are not meaningful when eager is enabled."""
        return self._handle

    @property
    def element_shape(self):
        return self._element_shape

    def identity(self):
        """See TensorArray."""
        return self.parent()

    def grad(self, source, flow=None, name=None):
        raise NotImplementedError("TensorArray.grad is not supported when executing eagerly; eager's gradient implementation does not use/need this function to compute gradients of operations that use TensorArrays.")

    def read(self, index, name=None):
        """See TensorArray."""
        del name
        if isinstance(index, ops.EagerTensor):
            index = index.numpy()
        if index < 0:
            raise errors_impl.OutOfRangeError(None, None, 'Reading from negative indices (index %d) is not allowed.' % index)
        if index >= len(self._tensor_array):
            raise errors_impl.OutOfRangeError(None, None, 'Tried to read from index %d but array size is: %d ' % (index, len(self._tensor_array)))
        tensor = self._tensor_array[index]
        if tensor is None:
            if index in self._previously_read_indices:
                raise errors_impl.InvalidArgumentError(None, None, 'Could not read index %d twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?)' % index)
            else:
                tensor = self._maybe_zero(index)
        if self._clear_after_read:
            self._tensor_array[index] = None
            self._previously_read_indices.append(index)
        return tensor

    def _write(self, index, value):
        """Writes `value` into index named by `index`.

    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The `Tensor` to write to `index`.

    Raises:
      errors_impl.InvalidArgumentError: `value` dtype does not match dtype.
      errors_impl.OutOfRangeError: `index` is out of bounds.
      ValueError: shape of `value` is not consistent with inferred shape.
    """
        if isinstance(index, ops.EagerTensor):
            index = index.numpy()
        if index < 0:
            raise errors_impl.OutOfRangeError(None, None, 'Writing to negative indices (index %d) is not allowed.' % index)
        size = len(self._tensor_array)
        if index >= size:
            if not self._dynamic_size:
                raise errors_impl.OutOfRangeError(None, None, 'Tried to write to index %d but array is not resizeable and size is: %d ' % (index, size))
            self._tensor_array.extend((None for _ in range(index - size + 1)))
        if not isinstance(value, ops.EagerTensor):
            value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
        if self._dtype != value.dtype:
            raise errors_impl.InvalidArgumentError(None, None, 'TensorArray dtype is %s but Op is trying to write dtype %s ' % (self._dtype.name, value.dtype.name))
        if not self._element_shape.is_compatible_with(value.shape):
            raise ValueError('Incompatible shape for value (%s), expected (%s)' % (value.shape, self._element_shape))
        if self._infer_shape:
            self._element_shape = self._element_shape.merge_with(value.shape)
        self._tensor_array[index] = value

    def write(self, index, value, name=None):
        """See TensorArray."""
        del name
        self._write(index, value)
        return self.parent()

    def _maybe_zero(self, ix):
        val = self._tensor_array[ix]
        if val is None:
            val = self._tensor_array[ix] = array_ops.zeros(shape=self._element_shape, dtype=self._dtype)
        return val

    def stack(self, name=None):
        """See TensorArray."""
        if self._tensor_array:
            for ix in range(len(self._tensor_array)):
                self._maybe_zero(ix)
        if not self._tensor_array and self._element_shape.is_fully_defined():
            return ops.convert_to_tensor(np.ndarray([0] + self._element_shape), name=name, dtype=self._dtype)
        else:
            return ops.convert_to_tensor(self._tensor_array, name=name, dtype=self._dtype)

    def gather(self, indices, name=None):
        """See TensorArray."""
        del name
        if isinstance(indices, ops.EagerTensor):
            indices = indices.numpy()
        return array_ops_stack.stack([self._maybe_zero(i) for i in indices])

    def concat(self, name=None):
        """See TensorArray."""
        try:
            return array_ops.concat([self._maybe_zero(ix) for ix in range(len(self._tensor_array))], 0, name=name)
        except errors_impl.OpError:
            shapes = [t.shape for t in self._tensor_array]
            ndims = [s.ndims for s in shapes]
            if 0 in ndims:
                idx = ndims.index(0)
                raise errors_impl.InvalidArgumentError(None, None, 'Concat saw a scalar shape at index %d but requires at least vectors.' % idx)
            else:
                raise

    def unstack(self, value, name=None):
        """See TensorArray."""
        tensors = array_ops_stack.unstack(value, name=name)
        if len(tensors) > len(self._tensor_array) and (not self._dynamic_size):
            raise ValueError('Cannot unstack %d tensors into a TensorArray of static size %d ' % (len(tensors), len(self._tensor_array)))
        self._tensor_array = tensors
        return self.parent()

    def scatter(self, indices, value, name=None):
        """See TensorArray."""
        del name
        if isinstance(indices, ops.EagerTensor):
            indices = indices.numpy()
        for index, val in zip(indices, array_ops_stack.unstack(value)):
            self._write(index, val)
        return self.parent()

    def split(self, value, lengths, name=None):
        """See TensorArray."""
        value = ops.convert_to_tensor(value, preferred_dtype=self._dtype, name='value')
        _check_dtypes(value, self._dtype)
        lengths = ops.convert_to_tensor(lengths)
        sum_lengths = math_ops.reduce_sum(lengths)
        if lengths.shape.ndims != 1:
            raise errors_impl.InvalidArgumentError(None, None, 'Expected lengths to be a vector, received shape: %s ' % lengths.shape.as_list())
        elif value.shape.ndims == 0:
            raise errors_impl.InvalidArgumentError(None, None, 'Expected value to be at least a vector, but received shape: %s ' % value.shape.as_list())
        elif sum_lengths.numpy() != value.shape.as_list()[0]:
            raise errors_impl.InvalidArgumentError(None, None, "Expected sum of lengths to be equal to values.shape[0], but sum of lengths is %d and value's shape is: %s " % (sum_lengths.numpy(), value.shape.as_list()))
        elif not self._dynamic_size and lengths.shape[0] != len(self._tensor_array):
            raise errors_impl.InvalidArgumentError(None, None, "TensorArray's size is not equal to the size of lengths (%d vs. %d), and the TensorArray is not marked as dynamically resizeable." % (len(self._tensor_array), lengths.shape[0]))
        else:
            self._tensor_array = array_ops.split(value, lengths, name=name)
            return self.parent()

    def size(self, name=None):
        """See TensorArray."""
        del name
        return constant_op.constant(len(self._tensor_array))

    def close(self, name=None):
        del name
        del self._tensor_array[:]