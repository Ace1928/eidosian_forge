from sympy.core.numbers import Integer
from sympy.matrices.dense import (eye, zeros)
def timeit_Matrix_zeronm():
    zeros(100, 100)