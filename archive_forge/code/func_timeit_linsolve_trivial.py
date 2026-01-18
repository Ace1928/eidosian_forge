from sympy.core.symbol import Symbol
from sympy.matrices.dense import (eye, zeros)
from sympy.solvers.solvers import solve_linear_system
def timeit_linsolve_trivial():
    solve_linear_system(M, *S)