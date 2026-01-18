import numpy as np
import pytest
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from cirq import Circuit, LineQubit
from cirq_rigetti.quil_input import (
from cirq.ops import (
def test_undefined_quil_gate():
    """There are no such things as FREDKIN & TOFFOLI in Quil. The standard
    names for those gates in Quil are CSWAP and CCNOT. Of course, they can
    be defined via DEFGATE / DEFCIRCUIT."""
    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil('FREDKIN 0 1 2')
    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil('TOFFOLI 0 1 2')