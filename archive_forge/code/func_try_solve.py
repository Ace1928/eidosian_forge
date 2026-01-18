import logging
from typing import Dict, Optional, Tuple, Type
import sympy
from torch.utils._sympy.functions import FloorDiv
def try_solve(expr: sympy.Basic, thing: sympy.Basic, trials: int=5, floordiv_inequality: bool=True) -> Optional[Tuple[sympy.Rel, sympy.Basic]]:
    mirror = mirror_rel_op(type(expr))
    if not isinstance(expr, sympy.Rel) or mirror is None:
        log.debug('expression with unsupported type: %s', type(expr))
        return None
    lhs_has_thing = expr.lhs.has(thing)
    rhs_has_thing = expr.rhs.has(thing)
    if lhs_has_thing and rhs_has_thing:
        log.debug('thing (%s) found in both sides of expression: %s', thing, expr)
        return None
    expressions = []
    if lhs_has_thing:
        expressions.append(expr)
    if rhs_has_thing:
        expressions.append(mirror(expr.rhs, expr.lhs))
    for e in expressions:
        if e is None:
            continue
        assert isinstance(e, sympy.Rel)
        for _ in range(trials):
            trial = _try_isolate_lhs(e, thing, floordiv_inequality=floordiv_inequality)
            if trial == e:
                break
            e = trial
        if isinstance(e, sympy.Rel) and e.lhs == thing:
            return (e, e.rhs)
    return None