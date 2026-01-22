from cupy._core import _fusion_variable
from cupy._core import _fusion_op
def _reduce_memory_access(ops):
    required_memories = set()
    for op in ops:
        for p in op.in_params + op.out_params:
            if p.memory.is_inout:
                required_memories.add(p.memory)
    for op in ops[::-1]:
        in_memories = set([p.memory for p in op.in_params])
        new_out_params = []
        for p in op.out_params:
            if p.memory in required_memories:
                new_out_params.append(p)
        op.out_params = _fusion_variable._VariableSet(*new_out_params)
        required_memories |= in_memories
    return [op for op in ops if len(op.out_params) > 0]