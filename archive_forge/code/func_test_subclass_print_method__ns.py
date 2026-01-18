from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p
def test_subclass_print_method__ns():

    class MyPrinter(CXX11CodePrinter):
        _ns = 'my_library::'
    p = CXX11CodePrinter()
    myp = MyPrinter()
    assert p.doprint(log1p(x)) == 'std::log1p(x)'
    assert myp.doprint(log1p(x)) == 'my_library::log1p(x)'