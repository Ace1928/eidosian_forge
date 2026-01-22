import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def _exponentiate_general_case(pauli_term: PauliTerm, param: float) -> Program:
    """
    Returns a Quil (Program()) object corresponding to the exponential of
    the pauli_term object, i.e. exp[-1.0j * param * pauli_term]

    :param pauli_term: A PauliTerm to exponentiate
    :param param: scalar, non-complex, value
    :returns: A Quil program object
    """

    def reverse_hack(p: Program) -> Program:
        revp = Program()
        revp.inst(list(reversed(p.instructions)))
        return revp
    quil_prog = Program()
    change_to_z_basis = Program()
    change_to_original_basis = Program()
    cnot_seq = Program()
    prev_index = None
    highest_target_index = None
    for index, op in pauli_term:
        assert isinstance(index, (int, QubitPlaceholder))
        if 'X' == op:
            change_to_z_basis.inst(H(index))
            change_to_original_basis.inst(H(index))
        elif 'Y' == op:
            change_to_z_basis.inst(RX(np.pi / 2.0, index))
            change_to_original_basis.inst(RX(-np.pi / 2.0, index))
        elif 'I' == op:
            continue
        if prev_index is not None:
            cnot_seq.inst(CNOT(prev_index, index))
        prev_index = index
        highest_target_index = index
    quil_prog += change_to_z_basis
    quil_prog += cnot_seq
    assert isinstance(pauli_term.coefficient, Complex) and highest_target_index is not None
    quil_prog.inst(RZ(2.0 * pauli_term.coefficient * param, highest_target_index))
    quil_prog += reverse_hack(cnot_seq)
    quil_prog += change_to_original_basis
    return quil_prog