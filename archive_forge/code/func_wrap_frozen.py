import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def wrap_frozen(*ops):
    return cirq.FrozenCircuit(wrap(*ops))