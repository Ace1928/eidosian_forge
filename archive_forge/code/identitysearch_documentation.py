from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
Returns the string of gates in a tuple.