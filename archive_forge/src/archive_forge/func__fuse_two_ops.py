from cupy._core import _fusion_variable
from cupy._core import _fusion_op
def _fuse_two_ops(op1, op2):
    """Returns a fused Op if the two ops can be fused, and ``None`` otherwise.
    """
    if not isinstance(op1, _fusion_op._ElementwiseTraceOp):
        return None
    if not isinstance(op2, _fusion_op._ElementwiseTraceOp):
        return None
    if op1.ashape != op2.ashape:
        return None
    new_in_params = op1.in_params + (op2.in_params - op1.out_params)
    new_out_params = op1.out_params + op2.out_params
    for in_param in new_in_params:
        for out_param in new_out_params:
            if in_param.memory == out_param.memory and in_param != out_param:
                return None
    op1.ops.extend(op2.ops)
    op1.in_params = new_in_params
    op1.out_params = new_out_params
    return op1