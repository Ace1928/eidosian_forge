from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_str_printing():
    assert vsprint(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'
    assert vsprint(omega.diff() * N.x) == "omega'*N.x"
    assert vsstrrepr(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'