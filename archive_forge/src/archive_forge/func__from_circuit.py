from __future__ import annotations
import re
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli, _count_y
@classmethod
def _from_circuit(cls, instr):
    """Convert a Pauli circuit to BasePauli data."""
    if isinstance(instr, (PauliGate, IGate, XGate, YGate, ZGate)):
        return cls._from_pauli_instruction(instr)
    if isinstance(instr, Instruction):
        if instr.definition is None:
            raise QiskitError(f'Cannot apply Instruction: {instr.name}')
        instr = instr.definition
    ret = Pauli(BasePauli(np.zeros((1, instr.num_qubits), dtype=bool), np.zeros((1, instr.num_qubits), dtype=bool), np.zeros(1, dtype=int)))
    if instr.global_phase:
        ret.phase = cls._phase_from_complex(np.exp(1j * float(instr.global_phase)))
    for inner in instr.data:
        if inner.clbits:
            raise QiskitError(f'Cannot apply instruction with classical bits: {inner.operation.name}')
        if not isinstance(inner.operation, (Barrier, Delay)):
            next_instr = BasePauli(*cls._from_circuit(inner.operation))
            if next_instr is not None:
                qargs = [instr.find_bit(tup).index for tup in inner.qubits]
                ret = ret.compose(next_instr, qargs=qargs)
    return (ret._z, ret._x, ret._phase)