from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _embedding_lookup_and_transform(params, ids, partition_strategy='mod', name=None, max_norm=None, transform_fn=None):
    """Helper function for embedding_lookup and _compute_sampled_logits.

  This function is a generalization of embedding_lookup that optionally
  applies a caller-specified transformation to each embedding. This is
  done through the `transform_fn` argument. If provided, the function is
  applied to each partitioned tensor of retrieved embeddings, colocated
  with the embeddings. This function will be called with a single `Tensor`
  argument of the same type as the `params` tensor and should return a
  `Tensor`. The shape of the argument will be the same as `params` except
  for the size of the first dimension. The first dimension of the result's
  shape must be the same size as the argument's.

  Args:
    params: See embedding_lookup.
    ids: See embedding_lookup.
    partition_strategy: See embedding_lookup.
    name: See embedding_lookup.
    max_norm: See embedding_lookup.
    transform_fn: An optional function to apply to each retrieved embedding. If
      max_norm is provided, transform_fn is applied to the norm-limited
      embeddings.

  Returns:
    See embedding_lookup for details.
  Raises:
    ValueError: If `params` is empty.
  """
    if params is None:
        raise ValueError('params must be specified')
    if isinstance(params, (list, tuple)) and (not params):
        raise ValueError('Length of params is currently 0. Need at least one param.')
    if isinstance(params, variables.PartitionedVariable):
        params = list(params)
    if not isinstance(params, list):
        params = [params]
    with ops.name_scope(name, 'embedding_lookup', params + [ids]) as name:
        np = len(params)
        if not any((isinstance(p, resource_variable_ops.BaseResourceVariable) for p in params)):
            params = indexed_slices.convert_n_to_tensor_or_indexed_slices(params, name='params')
        ids = ops.convert_to_tensor(ids, name='ids')
        if np == 1 and (not transform_fn or ids.get_shape().ndims == 1):
            with _colocate_with(params[0]):
                result = _clip(array_ops.gather(params[0], ids, name=name), ids, max_norm)
                if transform_fn:
                    result = transform_fn(result)
            return array_ops.identity(result)
        else:
            flat_ids = array_ops.reshape(ids, [-1])
            original_indices = math_ops.range(array_ops.size(flat_ids))
            if partition_strategy == 'mod':
                p_assignments = flat_ids % np
                new_ids = flat_ids // np
            elif partition_strategy == 'div':
                dim_0_size = tensor_shape.Dimension(tensor_shape.dimension_value(params[0].get_shape()[0]))
                for p in range(1, np):
                    dim_0_size += tensor_shape.Dimension(tensor_shape.dimension_value(params[p].get_shape()[0]))
                if dim_0_size.value:
                    num_total_ids = constant_op.constant(dim_0_size.value, flat_ids.dtype)
                else:
                    dim_0_sizes = []
                    for p in range(np):
                        param_p_dim = tensor_shape.dimension_value(params[p].get_shape()[0])
                        if param_p_dim is not None:
                            dim_0_sizes.append(param_p_dim)
                        else:
                            with _colocate_with(params[p]):
                                dim_0_sizes.append(array_ops.shape(params[p])[0])
                    num_total_ids = math_ops.reduce_sum(math_ops.cast(array_ops_stack.stack(dim_0_sizes), flat_ids.dtype))
                ids_per_partition = num_total_ids // np
                extras = num_total_ids % np
                p_assignments = math_ops.maximum(flat_ids // (ids_per_partition + 1), (flat_ids - extras) // ids_per_partition)
                new_ids = array_ops.where(p_assignments < extras, flat_ids % (ids_per_partition + 1), (flat_ids - extras) % ids_per_partition)
            else:
                raise ValueError(f'Unrecognized partition strategy: {partition_strategy}.Must be one of either `mod` or `div`.')
            p_assignments = math_ops.cast(p_assignments, dtypes.int32)
            gather_ids = data_flow_ops.dynamic_partition(new_ids, p_assignments, np)
            pindices = data_flow_ops.dynamic_partition(original_indices, p_assignments, np)
            partitioned_result = []
            for p in range(np):
                pids = gather_ids[p]
                with ops.device_v2(None):
                    with _colocate_with(params[p]):
                        result = array_ops.gather(params[p], pids)
                        if transform_fn:
                            result = transform_fn(_clip(result, pids, max_norm))
                partitioned_result.append(result)
            ret = data_flow_ops.parallel_dynamic_stitch(pindices, partitioned_result, name=name)
            if transform_fn is None:
                element_shape_s = params[0].get_shape()[1:]
                for p in params[1:]:
                    element_shape_s = element_shape_s.merge_with(p.get_shape()[1:])
            else:
                element_shape_s = ret.get_shape()[1:]
            if element_shape_s.is_fully_defined():
                element_shape_d = element_shape_s
            elif transform_fn is None:
                with _colocate_with(params[0]):
                    params_shape = array_ops.shape(params[0])
                element_shape_d = params_shape[1:]
            else:
                element_shape_d = array_ops.shape(ret)[1:]
            ret = array_ops.reshape(ret, array_ops.concat([array_ops.shape(ids), element_shape_d], 0))
            ret.set_shape(ids.get_shape().concatenate(element_shape_s))
            if not transform_fn:
                ret = _clip(ret, ids, max_norm)
            return ret