from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _isolate_zero(f, K, inf, sup, basis=False, sqf=False):
    """Handle special case of CF algorithm when ``f`` is homogeneous. """
    j, f = dup_terms_gcd(f, K)
    if j > 0:
        F = K.get_field()
        if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
            if not sqf:
                if not basis:
                    return ([((F.zero, F.zero), j)], f)
                else:
                    return ([((F.zero, F.zero), j, [K.one, K.zero])], f)
            else:
                return ([(F.zero, F.zero)], f)
    return ([], f)