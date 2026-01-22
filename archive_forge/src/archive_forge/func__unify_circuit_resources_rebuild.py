from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
def _unify_circuit_resources_rebuild(circuits: Tuple[QuantumCircuit, ...]) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """
    Ensure that all the given circuits have all the same qubits and clbits, and that they
    are defined in the same order.  The order is important for binding when the bodies are used in
    the 3-tuple :obj:`.Instruction` context.

    This function will always rebuild the objects into new :class:`.QuantumCircuit` instances.
    """
    from qiskit.circuit import QuantumCircuit
    qubits, clbits = (set(), set())
    for circuit in circuits:
        qubits.update(circuit.qubits)
        clbits.update(circuit.clbits)
    qubits, clbits = (list(qubits), list(clbits))
    out_circuits = []
    for circuit in circuits:
        out = QuantumCircuit(qubits, clbits, *circuit.qregs, *circuit.cregs, global_phase=circuit.global_phase)
        for instruction in circuit.data:
            out._append(instruction)
        out_circuits.append(out)
    return _unify_circuit_registers(out_circuits)