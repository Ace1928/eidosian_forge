import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _imul_atom_helper(self, key: TKey, pauli_lhs: int, sign: int) -> int:
    pauli_old = self.pauli_int_dict.pop(key, 0)
    pauli_new = pauli_lhs ^ pauli_old
    if pauli_new:
        self.pauli_int_dict[key] = pauli_new
    if not pauli_lhs or not pauli_old or pauli_lhs == pauli_old:
        return 0
    if (pauli_old - pauli_lhs) % 3 == 1:
        return sign
    return -sign