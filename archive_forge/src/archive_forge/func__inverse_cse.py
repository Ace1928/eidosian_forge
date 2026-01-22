from .. import Backend
import pytest
def _inverse_cse(subs_cses, cse_exprs):
    subs = dict(subs_cses)
    return [expr.subs(subs) for expr in cse_exprs]