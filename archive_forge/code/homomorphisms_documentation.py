from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed

        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a
        submodule of ``F``, and ``Q`` a submodule of ``S``, such that
        ``module = S/Q``, and ``c`` is a conversion function.
        