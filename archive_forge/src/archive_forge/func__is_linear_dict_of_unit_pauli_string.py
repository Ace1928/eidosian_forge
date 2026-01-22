from collections import defaultdict
from typing import (
import numbers
import numpy as np
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from scipy.sparse import csr_matrix
from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms
def _is_linear_dict_of_unit_pauli_string(linear_dict: value.LinearDict[UnitPauliStringT]) -> bool:
    if not isinstance(linear_dict, value.LinearDict):
        return False
    for k in linear_dict.keys():
        if not isinstance(k, frozenset):
            return False
        for qid, pauli in k:
            if not isinstance(qid, raw_types.Qid):
                return False
            if not isinstance(pauli, pauli_gates.Pauli):
                return False
    return True