from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def union_leaders(self, index1, index2):
    """Union in DSU."""
    leader1 = self.find_leader(index1)
    leader2 = self.find_leader(index2)
    if leader1 == leader2:
        return
    if len(self.group[leader1]) < len(self.group[leader2]):
        leader1, leader2 = (leader2, leader1)
    self.leader[leader2] = leader1
    self.group[leader1].extend(self.group[leader2])
    self.group[leader2].clear()