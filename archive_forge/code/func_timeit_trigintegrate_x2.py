from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.trigonometry import trigintegrate
def timeit_trigintegrate_x2():
    trigintegrate(x ** 2, x)