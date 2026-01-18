import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes
def try_to_reduce_precision(node, bounds, indirect_vars, indices, replacement_vals):

    def skip_filter(node):
        return node.target == 'to_dtype' and node.args[2] in (torch.int32, torch.float32, torch.float64)
    for dominated in dominated_nodes([node], skip_filter):
        if dominated.target in ['store', 'output']:
            continue
        if isinstance(dominated.target, str) and 'set_indirect' in dominated.target:
            idx = int(dominated.target[len('set_indirect'):])
            indirect_var = indirect_vars[idx]
            for index, expr in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]
                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return
                    index_val_int = ValueRanges(int(index_val.lower), int(index_val.upper))
                    if not range_expressable_in_32_bits(index_val_int):
                        return
        if not range_expressable_in_32_bits(bounds[dominated]):
            return
    args = list(node.args)
    args[2] = torch.int32
    node.args = tuple(args)