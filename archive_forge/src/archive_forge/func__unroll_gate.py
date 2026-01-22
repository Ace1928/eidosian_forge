from __future__ import annotations
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from ._utils import _ctrl_state_to_int
def _unroll_gate(operation, basis_gates):
    """Unrolls a gate, possibly composite, to the target basis"""
    circ = _gate_to_circuit(operation)
    pm = PassManager([UnrollCustomDefinitions(sel, basis_gates=basis_gates), BasisTranslator(sel, target_basis=basis_gates)])
    opqc = pm.run(circ)
    return opqc.to_gate()