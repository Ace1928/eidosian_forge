from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def all_gates_of_type(m: cirq.Moment, g: cirq.Gateset):
    for op in m:
        if op not in g:
            return False
    return True