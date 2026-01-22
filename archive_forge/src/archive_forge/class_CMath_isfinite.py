import cmath
from numba.core import types, utils
from numba.core.typing.templates import (AbstractTemplate, ConcreteTemplate,
@infer_global(cmath.isfinite)
class CMath_isfinite(CMath_predicate):
    pass