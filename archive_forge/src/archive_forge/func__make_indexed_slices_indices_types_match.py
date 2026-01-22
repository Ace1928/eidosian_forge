import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _make_indexed_slices_indices_types_match(op_type, branch_graphs):
    """Match dtype of IndexedSlices.indices in outputs of branch_graphs."""
    assert branch_graphs
    indexed_slice_indices = []
    current_index = 0
    branch_outputs_flat_with_composites = [nest.flatten(branch_graph.structured_outputs, expand_composites=False) for branch_graph in branch_graphs]
    outs_per_branch = [len(outs) for outs in branch_outputs_flat_with_composites]
    assert len(set(outs_per_branch)) == 1, outs_per_branch
    for output_idx, branch_outs in enumerate(zip(*branch_outputs_flat_with_composites)):
        if len(set((isinstance(out, indexed_slices.IndexedSlices) for out in branch_outs))) != 1:
            raise TypeError('Cannot reconcile tf.{op_name} {output_idx}-th outputs:\n  branches returned: {outputs}'.format(op_name='cond' if op_type == _COND else 'switch_case', output_idx=output_idx, outputs=branch_outs))
        if isinstance(branch_outs[0], indexed_slices.IndexedSlices):
            indexed_slice_indices.append(current_index + 1)
        if nest.is_nested_or_composite(branch_outs[0]):
            current_index += len(nest.flatten(branch_outs[0], expand_composites=True))
        elif branch_outs[0] is not None:
            current_index += 1
    if not indexed_slice_indices:
        return
    if current_index != len(branch_graphs[0].outputs):
        raise ValueError('Insufficient elements in branch_graphs[0].outputs.\nExpected: %i\nActual: %i' % (current_index, len(branch_graphs[0].outputs)))
    for index in indexed_slice_indices:
        if any((bg.outputs[index].dtype not in (dtypes.int32, dtypes.int64) for bg in branch_graphs)):
            raise TypeError('Type of IndexedSlices.indices must be int32 or int64. Found: %s' % str([bg.outputs[index].dtype for bg in branch_graphs]))
        if len(set((bg.outputs[index].dtype for bg in branch_graphs))) != 1:
            for branch_graph in branch_graphs:
                if branch_graph.outputs[index].dtype == dtypes.int32:
                    with branch_graph.as_default():
                        branch_graph.outputs[index] = math_ops.cast(branch_graph.outputs[index], dtypes.int64)
    for branch_graph in branch_graphs:
        branch_graph.structured_outputs = _pack_sequence_as(branch_graph.structured_outputs, branch_graph.outputs)