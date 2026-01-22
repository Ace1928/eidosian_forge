from math import pi
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import QuantumRegister, ControlFlowOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import (
@staticmethod
def _rxx_dag(parameter):
    _rxx_dag = DAGCircuit()
    qr = QuantumRegister(2)
    _rxx_dag.add_qreg(qr)
    _rxx_dag.apply_operation_back(RXXGate(parameter), [qr[1], qr[0]], [])
    return _rxx_dag