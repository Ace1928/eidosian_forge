from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
@cirq.value.value_equality(approximate=True, distinct_child_types=True)
class UnequalSycGate(cirq.FSimGate):

    def __init__(self, is_parameterized: bool=False):
        super().__init__(theta=THETA if is_parameterized else np.pi / 2, phi=PHI if is_parameterized else np.pi / 6)