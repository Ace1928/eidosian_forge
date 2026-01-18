from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.preview import preview
from io import BytesIO
def test_preview():
    x = Symbol('x')
    obj = BytesIO()
    try:
        preview(x, output='png', viewer='BytesIO', outputbuffer=obj)
    except RuntimeError:
        pass