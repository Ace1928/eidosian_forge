import cmath
from numba.core import types, utils
from numba.core.typing.templates import (AbstractTemplate, ConcreteTemplate,
@infer_global(cmath.log)
class Cmath_log(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in sorted(types.complex_domain)]
    cases += [signature(tp, tp, tp) for tp in sorted(types.complex_domain)]