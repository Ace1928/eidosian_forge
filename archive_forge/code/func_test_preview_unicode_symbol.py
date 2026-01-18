from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.preview import preview
from io import BytesIO
def test_preview_unicode_symbol():
    a = Symbol('α')
    obj = BytesIO()
    try:
        preview(a, output='png', viewer='BytesIO', outputbuffer=obj)
    except RuntimeError:
        pass