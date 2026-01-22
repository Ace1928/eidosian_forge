import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class BaseStagingArea:
    """Base class for Staging Areas."""
    _identifier = 0
    _lock = threading.Lock()

    def __init__(self, dtypes, shapes=None, names=None, shared_name=None, capacity=0, memory_limit=0):
        if shared_name is None:
            self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
        elif isinstance(shared_name, str):
            self._name = shared_name
        else:
            raise ValueError(f'shared_name must be a string, got {shared_name}')
        self._dtypes = dtypes
        if shapes is not None:
            if len(shapes) != len(dtypes):
                raise ValueError('StagingArea shapes must be the same length as dtypes')
            self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
        else:
            self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
        if names is not None:
            if len(names) != len(dtypes):
                raise ValueError('StagingArea names must be the same length as dtypes')
            self._names = names
        else:
            self._names = None
        self._capacity = capacity
        self._memory_limit = memory_limit
        with ops.name_scope('%s_root' % self._name):
            self._coloc_op = control_flow_ops.no_op()

    @property
    def name(self):
        """The name of the staging area."""
        return self._name

    @property
    def dtypes(self):
        """The list of dtypes for each component of a staging area element."""
        return self._dtypes

    @property
    def shapes(self):
        """The list of shapes for each component of a staging area element."""
        return self._shapes

    @property
    def names(self):
        """The list of names for each component of a staging area element."""
        return self._names

    @property
    def capacity(self):
        """The maximum number of elements of this staging area."""
        return self._capacity

    @property
    def memory_limit(self):
        """The maximum number of bytes of this staging area."""
        return self._memory_limit

    def _check_put_dtypes(self, vals, indices=None):
        """Validate and convert `vals` to a list of `Tensor`s.

    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.

    If `vals` is a list, then the appropriate indices associated with the
    values must be provided.

    If it is a dictionary, the staging area must have been constructed with a
    `names` attribute and the dictionary keys must match the staging area names.
    `indices` will be inferred from the dictionary keys.
    If the staging area was constructed with a `names` attribute, `vals` must
    be a dictionary.

    Checks that the dtype and shape of each value matches that
    of the staging area.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      A (tensors, indices) tuple where `tensors` is a list of `Tensor` objects
      and `indices` is a list of indices associated with the tensors.

    Raises:
      ValueError: If `vals` or `indices` is invalid.
    """
        if isinstance(vals, dict):
            if not self._names:
                raise ValueError('Staging areas must have names to enqueue a dictionary')
            if not set(vals.keys()).issubset(self._names):
                raise ValueError(f'Keys in dictionary to put do not match names of staging area. Dictionary: {sorted(vals.keys())}Queue: {sorted(self._names)}')
            vals, indices, _ = zip(*[(vals[k], i, k) for i, k in enumerate(self._names) if k in vals])
        else:
            if self._names:
                raise ValueError('You must enqueue a dictionary in a staging area with names')
            if indices is None:
                raise ValueError('Indices must be supplied when inserting a list of tensors')
            if len(indices) != len(vals):
                raise ValueError(f"Number of indices {len(indices)} doesn't match number of values {len(vals)}")
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
                indices = [0]
        if not len(vals) <= len(self._dtypes):
            raise ValueError(f'Unexpected number of inputs {len(vals)} vs {len(self._dtypes)}')
        tensors = []
        for val, i in zip(vals, indices):
            dtype, shape = (self._dtypes[i], self._shapes[i])
            if val.dtype != dtype:
                raise ValueError(f'Datatypes do not match. Received val.dtype {str(val.dtype)} and dtype {str(dtype)}')
            val.get_shape().assert_is_compatible_with(shape)
            tensors.append(ops.convert_to_tensor(val, dtype=dtype, name='component_%d' % i))
        return (tensors, indices)

    def _create_device_transfers(self, tensors):
        """Encode inter-device transfers if the current device
    is not the same as the Staging Area's device.
    """
        if not isinstance(tensors, (tuple, list)):
            tensors = [tensors]
        curr_device_scope = control_flow_ops.no_op().device
        if curr_device_scope != self._coloc_op.device:
            tensors = [array_ops.identity(t) for t in tensors]
        return tensors

    def _get_return_value(self, tensors, indices):
        """Return the value to return from a get op.

    If the staging area has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.

    Args:
      tensors: List of tensors from the get op.
      indices: Indices of associated names and shapes

    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """
        tensors = self._create_device_transfers(tensors)
        for output, i in zip(tensors, indices):
            output.set_shape(self._shapes[i])
        if self._names:
            return {self._names[i]: t for t, i in zip(tensors, indices)}
        return tensors

    def _scope_vals(self, vals):
        """Return a list of values to pass to `name_scope()`.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      The values in vals as a list.
    """
        if isinstance(vals, (list, tuple)):
            return vals
        elif isinstance(vals, dict):
            return vals.values()
        else:
            return [vals]