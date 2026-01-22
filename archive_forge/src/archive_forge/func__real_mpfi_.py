import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _real_mpfi_(self, RIF):

    def eval_term(k, v):
        pr = prod([_to_RIF(t, RIF, self._embed_cache) for t in k], RIF(1))
        if not pr > 0:
            raise _SqrtException()
        return pr.sqrt() * _to_RIF(v, RIF, self._embed_cache)
    return sum([eval_term(k, v) for k, v in self._dict.items()], RIF(0))