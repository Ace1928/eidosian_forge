from math import pi
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import QuantumRegister, ControlFlowOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import (
@staticmethod
def _rzx_dag(parameter):
    _rzx_dag = DAGCircuit()
    qr = QuantumRegister(2)
    _rzx_dag.add_qreg(qr)
    _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
    _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
    _rzx_dag.apply_operation_back(RZXGate(parameter), [qr[1], qr[0]], [])
    _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
    _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
    return _rzx_dag