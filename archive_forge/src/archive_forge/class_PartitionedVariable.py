import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
class PartitionedVariable:
    """A container for partitioned `Variable` objects.

  @compatibility(eager) `tf.PartitionedVariable` is not compatible with
  eager execution.  Use `tf.Variable` instead which is compatible
  with both eager execution and graph construction.  See [the
  TensorFlow Eager Execution
  guide](https://www.tensorflow.org/guide/eager#variables_and_optimizers)
  for details on how variables work in eager execution.
  @end_compatibility
  """

    def __init__(self, name, shape, dtype, variable_list, partitions):
        """Creates a new partitioned variable wrapper.

    Variables passed via the variable_list must contain a save_slice_info
    field.  Concatenation and iteration is in lexicographic order according
    to the var_offset property of the save_slice_info.

    Args:
      name: String. Overall name of the variables.
      shape: List of integers.  Overall shape of the variables.
      dtype: Type of the variables.
      variable_list: List of `Variable` that comprise this partitioned variable.
      partitions: List of integers.  Number of partitions for each dimension.

    Raises:
      TypeError: If `variable_list` is not a list of `Variable` objects, or
        `partitions` is not a list.
      ValueError: If `variable_list` is empty, or the `Variable` shape
        information does not match `shape`, or `partitions` has invalid values.
    """
        if not isinstance(variable_list, (list, tuple)):
            raise TypeError('variable_list is not a list or tuple: %s' % variable_list)
        if not isinstance(partitions, (list, tuple)):
            raise TypeError('partitions is not a list or tuple: %s' % partitions)
        if not all((p >= 1 for p in partitions)):
            raise ValueError('partition values must be positive: %s' % partitions)
        if not variable_list:
            raise ValueError('variable_list may not be empty')
        for v in variable_list:
            if not all((v._get_save_slice_info() is not None for v in variable_list)):
                raise ValueError('All variables must have a save_slice_info available: %s' % [v.name for v in variable_list])
            if len(shape) != len(partitions):
                raise ValueError('len(shape) != len(partitions): %s vs. %s' % (shape, partitions))
            if v._get_save_slice_info().full_shape != shape:
                raise ValueError("All variables' full shapes must match shape: %s; but full shapes were: %s" % (shape, str([v._get_save_slice_info().full_shape])))
        self._variable_list = sorted(variable_list, key=lambda v: v._get_save_slice_info().var_offset)
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._partitions = partitions
        self._as_tensor = None

    def __iter__(self):
        """Return an iterable for accessing the underlying partition Variables."""
        return iter(self._variable_list)

    def __len__(self):
        num_partition_axes = len(self._partition_axes())
        if num_partition_axes > 1:
            raise ValueError('Cannot get a length for %d > 1 partition axes' % num_partition_axes)
        return len(self._variable_list)

    def _partition_axes(self):
        if all((p == 1 for p in self._partitions)):
            return [0]
        else:
            return [i for i, p in enumerate(self._partitions) if p > 1]

    def _concat(self):
        """Returns the overall concatenated value as a `Tensor`.

    This is different from using the partitioned variable directly as a tensor
    (through tensor conversion and `as_tensor`) in that it creates a new set of
    operations that keeps the control dependencies from its scope.

    Returns:
      `Tensor` containing the concatenated value.
    """
        if len(self._variable_list) == 1:
            with ops.name_scope(None):
                return array_ops.identity(self._variable_list[0], name=self._name)
        partition_axes = self._partition_axes()
        if len(partition_axes) > 1:
            raise NotImplementedError('Cannot concatenate along more than one dimension: %s.  Multi-axis partition concat is not supported' % str(partition_axes))
        partition_ix = partition_axes[0]
        with ops.name_scope(self._name + '/ConcatPartitions/'):
            concatenated = array_ops.concat(self._variable_list, partition_ix)
        with ops.name_scope(None):
            return array_ops.identity(concatenated, name=self._name)

    def as_tensor(self):
        """Returns the overall concatenated value as a `Tensor`.

    The returned tensor will not inherit the control dependencies from the scope
    where the value is used, which is similar to getting the value of
    `Variable`.

    Returns:
      `Tensor` containing the concatenated value.
    """
        with ops.control_dependencies(None):
            return self._concat()

    @staticmethod
    def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):
        _ = name
        if dtype is not None and (not dtype.is_compatible_with(v.dtype)):
            raise ValueError("Incompatible type conversion requested to type '%s' for variable of type '%s'" % (dtype.name, v.dtype.name))
        if as_ref:
            raise NotImplementedError("PartitionedVariable doesn't support being used as a reference.")
        else:
            return v.as_tensor()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.get_shape()

    @property
    def _distribute_strategy(self):
        """The `tf.distribute.Strategy` that this variable was created under."""
        return None

    def get_shape(self):
        return self._shape

    def _get_variable_list(self):
        return self._variable_list

    def _get_partitions(self):
        return self._partitions

    def _apply_assign_fn(self, assign_fn, value):
        partition_axes = self._partition_axes()
        if len(partition_axes) > 1:
            raise NotImplementedError('Cannot do assign action along more than one dimension: %s.  Multi-axis partition assign action is not supported ' % str(partition_axes))
        if isinstance(value, list):
            assert len(value) == len(self._variable_list)
            value_list = value
        elif isinstance(value, PartitionedVariable):
            value_list = list(value)
        else:
            partition_ix = partition_axes[0]
            size_splits_list = [tensor_shape.dimension_value(var.shape[partition_ix]) for var in self._variable_list]
            value_list = array_ops.split(value, size_splits_list, axis=partition_ix)
        op_list = [assign_fn(var, value_list[idx]) for idx, var in enumerate(self._variable_list)]
        return op_list

    def assign(self, value, use_locking=False, name=None, read_value=True):
        assign_fn = lambda var, r_value: var.assign(r_value, use_locking=use_locking, name=name, read_value=read_value)
        assign_list = self._apply_assign_fn(assign_fn, value)
        if read_value:
            return assign_list
        return [assign.op for assign in assign_list]

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        assign_fn = lambda var, r_value: var.assign_add(r_value, use_locking=use_locking, name=name, read_value=read_value)
        assign_list = self._apply_assign_fn(assign_fn, value)
        if read_value:
            return assign_list
        return [assign.op for assign in assign_list]

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        assign_fn = lambda var, r_value: var.assign_sub(r_value, use_locking=use_locking, name=name, read_value=read_value)
        assign_list = self._apply_assign_fn(assign_fn, value)
        if read_value:
            return assign_list
        return [assign.op for assign in assign_list]