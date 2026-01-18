from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
A tuple of the Hilbert spaces in this direct sum.