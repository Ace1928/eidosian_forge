import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _slice_helper(tensor, slice_spec, update_method=None, updates=None):
    """Helper function for __getitem__ and _with_index_update_helper.

  This function collects the indices in `slice_spec` into two buckets, which we
  can call "idx1" and "idx2" here. idx1 is intended for `strided_slice`, idx2
  `gather`.  They also correspond to "basic indices" and "advanced indices" in
  numpy.  This function supports both reading and writing at the indices. The
  reading path can be summarized as `gather(stride_slice(tensor, idx1),
  idx2)`. The writing path can be summarized as `strided_slice_update(tensor,
  idx1, scatter(strided_slice(tensor, idx1), idx2, updates))`.  (`gather` here
  means `tf.gather` or `tf.gather_nd`; `scatter` here means
  `tf.tensor_scatter_update`.)  The writing path is inefficient because it needs
  to first read out a portion (probably much larger than `updates`) of `tensor`
  using `strided_slice`, update it, and then write the portion back. An
  alternative approach is to only use `scatter`, which amounts to using the
  indexing mechanism of gather/scatter to implement
  strided_slice/strided_slice_update. This is feasible for XLA Gather/Scatter
  because they support spans (e.g. `2:5`) in indices (as begin/end pairs), but
  not TF gather/scatter because they don't support spans (except those that
  cover entire dimensions, i.e. `:`).  If we materialize spans into individual
  indices, the size of the index tensor would explode.  (Note that XLA
  Gather/Scatter have a similar problem for stride > 1 because they don't
  support strides.  Indices such as `1:2:8` will need to be materialized into
  individual indices such as [1, 3, 5, 7].)

  Args:
    tensor: the tensor to be read from or write into.
    slice_spec: the indices.
    update_method: (optional) a member of `_UpdateMethod`, indicating how to
      update the values (replacement, add, etc.). `None` indicates just reading.
    updates: (optional) the new values to write into `tensor`. It must have the
      same dtype as `tensor`.

  Returns:
    The result of reading (if `update_method` is `None`) or the updated `tensor`
    after writing.
  """
    begin, end, strides = ([], [], [])
    new_axis_mask, shrink_axis_mask = (0, 0)
    begin_mask, end_mask = (0, 0)
    ellipsis_mask = 0
    advanced_indices = []
    shrink_indices = []
    for index, s in enumerate(slice_spec):
        if isinstance(s, slice):
            if s.start is not None:
                begin.append(_as_index(s.start)[0])
            else:
                begin.append(0)
                begin_mask |= 1 << index
            if s.stop is not None:
                end.append(_as_index(s.stop)[0])
            else:
                end.append(0)
                end_mask |= 1 << index
            if s.step is not None:
                strides.append(_as_index(s.step)[0])
            else:
                strides.append(1)
        elif s is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= 1 << index
        elif s is array_ops.newaxis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            new_axis_mask |= 1 << index
        else:
            s, is_scalar = _as_index(s, False)
            if is_scalar:
                begin.append(s)
                end.append(s + 1)
                strides.append(1)
                shrink_axis_mask |= 1 << index
                shrink_indices.append(index)
            else:
                begin.append(0)
                end.append(0)
                strides.append(1)
                begin_mask |= 1 << index
                end_mask |= 1 << index
                advanced_indices.append((index, s, ellipsis_mask != 0))
    with ops.name_scope(None, 'strided_slice', [tensor] + begin + end + strides, skip_on_eager=False) as name:
        if begin:
            packed_begin, packed_end, packed_strides = (array_ops_stack.stack(begin), array_ops_stack.stack(end), array_ops_stack.stack(strides))
            if packed_begin.dtype == dtypes.int64 or packed_end.dtype == dtypes.int64 or packed_strides.dtype == dtypes.int64:
                if packed_begin.dtype != dtypes.int64:
                    packed_begin = math_ops.cast(packed_begin, dtypes.int64)
                if packed_end.dtype != dtypes.int64:
                    packed_end = math_ops.cast(packed_end, dtypes.int64)
                if packed_strides.dtype != dtypes.int64:
                    packed_strides = math_ops.cast(packed_strides, dtypes.int64)
        else:
            var_empty = constant_op.constant([], dtype=dtypes.int32)
            packed_begin = packed_end = packed_strides = var_empty
        if update_method == _UpdateMethod.UPDATE and (not advanced_indices):
            return array_ops.tensor_strided_slice_update(tensor, packed_begin, packed_end, packed_strides, updates, begin_mask=begin_mask, end_mask=end_mask, shrink_axis_mask=shrink_axis_mask, new_axis_mask=new_axis_mask, ellipsis_mask=ellipsis_mask, name=name)
        else:
            if updates is not None:
                original_tensor = tensor
            tensor = array_ops.strided_slice(tensor, packed_begin, packed_end, packed_strides, begin_mask=begin_mask, end_mask=end_mask, shrink_axis_mask=shrink_axis_mask, new_axis_mask=new_axis_mask, ellipsis_mask=ellipsis_mask, name=name)
        if not advanced_indices:
            if update_method is None:
                return tensor
            assert update_method != _UpdateMethod.UPDATE
            if update_method == _UpdateMethod.ADD:
                update_op = math_ops.add
            elif update_method == _UpdateMethod.MIN:
                update_op = math_ops.minimum
            elif update_method == _UpdateMethod.MAX:
                update_op = math_ops.maximum
            return array_ops.tensor_strided_slice_update(original_tensor, packed_begin, packed_end, packed_strides, update_op(tensor, updates), begin_mask=begin_mask, end_mask=end_mask, shrink_axis_mask=shrink_axis_mask, new_axis_mask=new_axis_mask, ellipsis_mask=ellipsis_mask, name=name + '_2')
        advanced_indices_map = {}
        for index, data, had_ellipsis in advanced_indices:
            if had_ellipsis:
                num_shrink = len([x for x in shrink_indices if x > index])
                dim = index - len(slice_spec) + num_shrink
            else:
                num_shrink = len([x for x in shrink_indices if x < index])
                dim = index - num_shrink
            advanced_indices_map[dim] = data
        dims = sorted(advanced_indices_map.keys())
        dims_contiguous = True
        if len(dims) > 1:
            if dims[0] < 0 and dims[-1] >= 0:
                dims_contiguous = False
            else:
                for i in range(len(dims) - 1):
                    if dims[i] + 1 != dims[i + 1]:
                        dims_contiguous = False
                        break
        indices = [advanced_indices_map[x] for x in dims]
        indices = _promote_dtype(*indices)
        indices = np_utils.tf_broadcast(*indices)
        stacked_indices = array_ops_stack.stack(indices, axis=-1)
        if not dims_contiguous or updates is not None:
            if range(len(dims)) != dims:
                tensor = moveaxis(tensor, dims, range(len(dims)))
            tensor_shape_prefix = array_ops.shape(tensor, out_type=stacked_indices.dtype)[:len(dims)]
            stacked_indices = array_ops.where_v2(stacked_indices < 0, stacked_indices + tensor_shape_prefix, stacked_indices)
            if updates is None:
                return array_ops.gather_nd(tensor, stacked_indices)
            else:
                if dims_contiguous:
                    if stacked_indices.shape.rank is None:
                        raise NotImplementedError('Rank of the advanced indices must currently be known')
                    batch_size = stacked_indices.shape.rank - 1
                    batch_start = dims[0]
                    if batch_start < 0:
                        batch_start += len(dims) - batch_size

                    def range_(start, length):
                        return range(start, start + length)
                    updates = moveaxis(updates, range_(batch_start, batch_size), range(batch_size))
                if update_method == _UpdateMethod.UPDATE:
                    update_op = array_ops.tensor_scatter_update
                elif update_method == _UpdateMethod.ADD:
                    update_op = array_ops.tensor_scatter_add
                elif update_method == _UpdateMethod.MIN:
                    update_op = array_ops.tensor_scatter_min
                elif update_method == _UpdateMethod.MAX:
                    update_op = array_ops.tensor_scatter_max
                tensor = update_op(tensor, stacked_indices, updates)
                if range(len(dims)) != dims:
                    tensor = moveaxis(tensor, range(len(dims)), dims)
                return array_ops.tensor_strided_slice_update(original_tensor, packed_begin, packed_end, packed_strides, tensor, begin_mask=begin_mask, end_mask=end_mask, shrink_axis_mask=shrink_axis_mask, new_axis_mask=new_axis_mask, ellipsis_mask=ellipsis_mask, name=name + '_2')
        rank = np_utils._maybe_static(array_ops.rank(tensor))
        dims = [x + rank if x < 0 else x for x in dims]
        shape_tensor = array_ops.shape(tensor)
        dim_sizes = array_ops.gather(shape_tensor, dims)
        if len(dims) == 1:
            stacked_indices = indices[0]
        stacked_indices = math_ops.cast(stacked_indices, dtypes.int32)
        stacked_indices = array_ops.where_v2(stacked_indices < 0, stacked_indices + dim_sizes, stacked_indices)
        axis = dims[0]
        if len(dims) > 1:
            index_scaling = math_ops.cumprod(dim_sizes, reverse=True, exclusive=True)

            def _tensordot(a, b):
                b = array_ops.broadcast_to(b, array_ops.shape(a))
                return math_ops.reduce_sum(a * b, axis=-1)
            stacked_indices = _tensordot(stacked_indices, index_scaling)
            flat_shape = array_ops.concat([shape_tensor[:axis], [-1], shape_tensor[axis + len(dims):]], axis=0)
            tensor = array_ops.reshape(tensor, flat_shape)
        return array_ops.gather(tensor, stacked_indices, axis=axis)