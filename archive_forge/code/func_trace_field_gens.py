from .shapes import polished_tetrahedra_shapes
from ..sage_helper import _within_sage, sage_method
from .polished_reps import polished_holonomy
from . import nsagetools, interval_reps, slice_obs_HKL
from .character_varieties import character_variety, character_variety_ideal
@sage_method
def trace_field_gens(manifold, fundamental_group_args=[]):
    """
    The generators of the trace field as ApproximateAlgebraicNumbers. Can be
    used to compute the tetrahedra field, where the first two parameters
    are bits of precision and maximum degree of the field::

        sage: M = Manifold('m125')
        sage: traces = M.trace_field_gens()
        sage: traces.find_field(100, 10, optimize=True)    # doctest: +NORMALIZE_WHITESPACE
        (Number Field in z with defining polynomial x^2 + 1
         with z = -1*I,
        <ApproxAN: -1.0*I>, [z + 1, z, z + 1])
    """

    def func(prec):
        return polished_holonomy(manifold, prec, fundamental_group_args).trace_field_generators()
    return ListOfApproximateAlgebraicNumbers(func)