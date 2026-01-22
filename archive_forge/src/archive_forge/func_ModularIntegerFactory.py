from __future__ import annotations
from typing import Any
import operator
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
def ModularIntegerFactory(_mod, _dom, _sym, parent):
    """Create custom class for specific integer modulus."""
    try:
        _mod = _dom.convert(_mod)
    except CoercionFailed:
        ok = False
    else:
        ok = True
    if not ok or _mod < 1:
        raise ValueError('modulus must be a positive integer, got %s' % _mod)
    key = (_mod, _dom, _sym)
    try:
        cls = _modular_integer_cache[key]
    except KeyError:

        class cls(ModularInteger):
            mod, dom, sym = (_mod, _dom, _sym)
            _parent = parent
        if _sym:
            cls.__name__ = 'SymmetricModularIntegerMod%s' % _mod
        else:
            cls.__name__ = 'ModularIntegerMod%s' % _mod
        _modular_integer_cache[key] = cls
    return cls