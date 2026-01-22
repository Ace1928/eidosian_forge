import warnings
from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from .entity import GeometryEntity
from mpmath.libmp.libmpf import prec_to_dps
@classmethod
def _normalize_dimension(cls, *points, **kwargs):
    """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
    dim = getattr(cls, '_ambient_dimension', None)
    dim = kwargs.get('dim', dim)
    if dim is None:
        dim = max((i.ambient_dimension for i in points))
    if all((i.ambient_dimension == dim for i in points)):
        return list(points)
    kwargs['dim'] = dim
    kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
    return [Point(i, **kwargs) for i in points]