import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def _to_RIF(x, RIF, embed_cache=None):
    """
    Given a Sage Integer, Rational or an element x in a
    Sage NumberField with a real embedding and an instance
    of a RealIntervalField to specify the desired precision,
    return a real interval containing the true value of x.

    Warning: one can actually call RIF(x) and get an interval, but I have
    found examples where that interval does not contain the true value!
    Seems a bug in Sage. CIF(x) doesn't work, so maybe there is just some
    sequence of casts going on to convert x to an interval that wasn't
    anticipated.
    """
    if isinstance(x, Integer) or isinstance(x, Rational):
        return RIF(x)
    nf = x.parent()
    root = _get_interval_embedding_from_cache(nf, RIF, embed_cache)
    if not nf.defining_polynomial()(root).contains_zero():
        raise Exception('Root failed test.')
    return x.lift()(root)