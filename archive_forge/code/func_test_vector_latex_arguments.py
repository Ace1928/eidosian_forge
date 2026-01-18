from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_vector_latex_arguments():
    assert vlatex(N.x * 3.0, full_prec=False) == '3.0\\mathbf{\\hat{n}_x}'
    assert vlatex(N.x * 3.0, full_prec=True) == '3.00000000000000\\mathbf{\\hat{n}_x}'