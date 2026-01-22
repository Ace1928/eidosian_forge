from __future__ import annotations
from typing import List, Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from .standard_gates import MCXGate
def _zero_reflection(num_state_qubits: int, qubits: List[int], mcx_mode: Optional[str]=None) -> QuantumCircuit:
    qr_state = QuantumRegister(num_state_qubits, 'state')
    reflection = QuantumCircuit(qr_state, name='S_0')
    num_ancillas = MCXGate.get_num_ancilla_qubits(len(qubits) - 1, mcx_mode)
    if num_ancillas > 0:
        qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
        reflection.add_register(qr_ancilla)
    else:
        qr_ancilla = AncillaRegister(0)
    reflection.x(qubits)
    if len(qubits) == 1:
        reflection.z(0)
    else:
        reflection.h(qubits[-1])
        reflection.mcx(qubits[:-1], qubits[-1], qr_ancilla[:], mode=mcx_mode)
        reflection.h(qubits[-1])
    reflection.x(qubits)
    return reflection