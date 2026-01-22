import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _get_embed_cache(l1, l2):
    """
    Given objects of type SqrtLinCombination or _FactorizedSqrtLinCombination
    return the first _embed_cache that is not None.
    For example, one SqrtLinCombination might be instantiated from an
    Integer and the other from an element in the number field that we are
    currently working in. Then only the latter one has an _embed_cache. Thus,
    the need for this function when adding, multiplying, ... those two
    instances.
    """
    for l in [l1, l2]:
        if (isinstance(l, SqrtLinCombination) or isinstance(l, _FactorizedSqrtLinCombination)) and l._embed_cache is not None:
            return l._embed_cache
    return None