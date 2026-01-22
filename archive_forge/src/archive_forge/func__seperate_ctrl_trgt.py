from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _seperate_ctrl_trgt(self, node):
    """Get the target qubits and control qubits if available,
        as well as their respective z3 variables.
        """
    gate = node.op
    if isinstance(gate, ControlledGate):
        numctrl = gate.num_ctrl_qubits
    else:
        numctrl = 0
    ctrlqb = node.qargs[:numctrl]
    trgtqb = node.qargs[numctrl:]
    try:
        ctrlvar = [self.variables[qb][self.varnum[qb][node]] for qb in ctrlqb]
        trgtvar = [self.variables[qb][self.varnum[qb][node]] for qb in trgtqb]
    except KeyError:
        ctrlvar = [self.variables[qb][-1] for qb in ctrlqb]
        trgtvar = [self.variables[qb][-1] for qb in trgtqb]
    return (ctrlqb, ctrlvar, trgtqb, trgtvar)