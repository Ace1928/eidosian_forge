from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyerrors import CoercionFailed, NotReversible, NotInvertible
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.factortools import (
from sympy.polys.rootisolation import (
from sympy.polys.polyerrors import (
def to_sympy_dict(f):
    """Convert ``f`` to a dict representation with SymPy coefficients. """
    rep = dmp_to_dict(f.rep, 0, f.dom)
    for k, v in rep.items():
        rep[k] = f.dom.to_sympy(v)
    return rep