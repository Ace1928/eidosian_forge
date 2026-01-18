from .shapes import polished_tetrahedra_shapes
from ..sage_helper import _within_sage, sage_method
from .polished_reps import polished_holonomy
from . import nsagetools, interval_reps, slice_obs_HKL
from .character_varieties import character_variety, character_variety_ideal
@sage_method
def tetrahedra_field_gens(manifold):
    """
    The shapes of the tetrahedra as ApproximateAlgebraicNumbers. Can be
    used to compute the tetrahedra field, where the first two parameters
    are bits of precision and maximum degree of the field::

        sage: M = Manifold('m015')
        sage: tets = M.tetrahedra_field_gens()
        sage: tets.find_field(100, 10, optimize=True)    # doctest: +NORMALIZE_WHITESPACE +NUMERIC9
        (Number Field in z with defining polynomial x^3 - x - 1
         with z = -0.6623589786223730? - 0.5622795120623013?*I,
        <ApproxAN: -0.662358978622 - 0.562279512062*I>, [-z, -z, -z])
    """
    if manifold.is_orientable():

        def func(prec):
            return polished_tetrahedra_shapes(manifold, bits_prec=prec)
    else:
        double_cover = manifold.orientation_cover()

        def func(prec):
            return polished_tetrahedra_shapes(double_cover, bits_prec=prec)[::2]
    return ListOfApproximateAlgebraicNumbers(func)