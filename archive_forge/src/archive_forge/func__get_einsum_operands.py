import warnings
import cupy
from cupy._core import _accelerator
def _get_einsum_operands(args):
    """Parse & retrieve einsum operands, assuming ``args`` is in either
    "subscript" or "interleaved" format.
    """
    if len(args) == 0:
        raise ValueError('must specify the einstein sum subscripts string and at least one operand, or at least one operand and its corresponding subscripts list')
    if isinstance(args[0], str):
        expr = args[0]
        operands = list(args[1:])
        return (expr, operands)
    else:
        args = list(args)
        operands = []
        inputs = []
        output = None
        while len(args) >= 2:
            operands.append(args.pop(0))
            inputs.append(args.pop(0))
        if len(args) == 1:
            output = args.pop(0)
        assert not args
        return (inputs, operands, output)