from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterTable, ParameterView
def to_instruction(self, parameter_map=None, label=None):
    if not self._is_built:
        self._build()
    return super().to_instruction(parameter_map, label=label)