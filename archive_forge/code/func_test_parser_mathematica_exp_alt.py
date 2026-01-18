from sympy import sin, Function, symbols, Dummy, Lambda, cos
from sympy.parsing.mathematica import parse_mathematica, MathematicaParser
from sympy.core.sympify import sympify
from sympy.abc import n, w, x, y, z
from sympy.testing.pytest import raises
def test_parser_mathematica_exp_alt():
    parser = MathematicaParser()
    convert_chain2 = lambda expr: parser._from_fullformlist_to_fullformsympy(parser._from_fullform_to_fullformlist(expr))
    convert_chain3 = lambda expr: parser._from_fullformsympy_to_sympy(convert_chain2(expr))
    Sin, Times, Plus, Power = symbols('Sin Times Plus Power', cls=Function)
    full_form1 = 'Sin[Times[x, y]]'
    full_form2 = 'Plus[Times[x, y], z]'
    full_form3 = 'Sin[Times[x, Plus[y, z], Power[w, n]]]]'
    assert parser._from_fullform_to_fullformlist(full_form1) == ['Sin', ['Times', 'x', 'y']]
    assert parser._from_fullform_to_fullformlist(full_form2) == ['Plus', ['Times', 'x', 'y'], 'z']
    assert parser._from_fullform_to_fullformlist(full_form3) == ['Sin', ['Times', 'x', ['Plus', 'y', 'z'], ['Power', 'w', 'n']]]
    assert convert_chain2(full_form1) == Sin(Times(x, y))
    assert convert_chain2(full_form2) == Plus(Times(x, y), z)
    assert convert_chain2(full_form3) == Sin(Times(x, Plus(y, z), Power(w, n)))
    assert convert_chain3(full_form1) == sin(x * y)
    assert convert_chain3(full_form2) == x * y + z
    assert convert_chain3(full_form3) == sin(x * (y + z) * w ** n)