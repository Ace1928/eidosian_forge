import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer = True

    @classmethod
    def eval(cls, *args):
        assert len(args) % 2 == 0
        dim = len(args) // 2
        if all((isinstance(a, sympy.Integer) for a in args)):
            from torch.fx.experimental.symbolic_shapes import eval_is_non_overlapping_and_dense
            size_args = args[0:dim]
            stride_args = args[dim:]
            return eval_is_non_overlapping_and_dense([int(a) for a in size_args], [int(a) for a in stride_args])
        return None