import collections
from llvmlite.ir import context, values, types, _utils
def _fix_di_operands(self, operands):
    fixed_ops = []
    for name, op in operands:
        if isinstance(op, (list, tuple)):
            op = self.add_metadata(op)
        fixed_ops.append((name, op))
    return fixed_ops