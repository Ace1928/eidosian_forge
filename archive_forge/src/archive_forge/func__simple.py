import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _simple(p):
    if len(p) != 1:
        return False
    op, av = p[0]
    if op is SUBPATTERN:
        return av[0] is None and _simple(av[-1])
    return op in _UNIT_CODES