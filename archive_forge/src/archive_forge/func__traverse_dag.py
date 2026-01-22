from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _traverse_dag(self, dag):
    """traverse DAG in topological order
            for each gate check: if any control is 0, or
                                 if triviality conditions are satisfied
            if yes remove gate from dag
            apply postconditions of gate
        Args:
            dag (DAGCircuit): input DAG to optimize in place
        """
    import z3
    for node in dag.topological_op_nodes():
        gate = node.op
        ctrlqb, ctrlvar, trgtqb, trgtvar = self._seperate_ctrl_trgt(node)
        ctrl_ones = z3.And(*ctrlvar)
        remove_ctrl, new_dag, qb_idx = self._remove_control(gate, ctrlvar, trgtvar)
        if remove_ctrl:
            dag.substitute_node_with_dag(node, new_dag)
            gate = gate.base_gate
            node.op = gate.to_mutable()
            node.name = gate.name
            node.qargs = tuple(((ctrlqb + trgtqb)[qi] for qi in qb_idx))
            _, ctrlvar, trgtqb, trgtvar = self._seperate_ctrl_trgt(node)
            ctrl_ones = z3.And(*ctrlvar)
        trivial = self._test_gate(gate, ctrl_ones, trgtvar)
        if trivial:
            dag.remove_op_node(node)
        elif self.size > 1:
            for qbt in node.qargs:
                self.gatecache[qbt].append(node)
                self.varnum[qbt][node] = self.gatenum[qbt] - 1
            for qbt in node.qargs:
                if len(self.gatecache[qbt]) >= self.size:
                    self._multigate_opt(dag, qbt)
        self._add_postconditions(gate, ctrl_ones, trgtqb, trgtvar)