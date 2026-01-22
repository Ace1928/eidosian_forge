from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
from sympy.polys.polyerrors import CoercionFailed
def _check_hom(self, oth):
    """Helper to check that oth is a homomorphism with same domain/codomain."""
    if not isinstance(oth, ModuleHomomorphism):
        return False
    return oth.domain == self.domain and oth.codomain == self.codomain