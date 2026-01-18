from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
def union_set(self, set1, set2):
    """DSU function for unioning two sets together
        Find the roots of each set. Then assign one to have the other
        as its parent, thus liking the sets.
        Merges smaller set into larger set in order to have better runtime
        """
    set1 = self.find_set(set1)
    set2 = self.find_set(set2)
    if set1 == set2:
        return
    if len(self.gate_groups[set1]) < len(self.gate_groups[set2]):
        set1, set2 = (set2, set1)
    self.parent[set2] = set1
    self.gate_groups[set1].extend(self.gate_groups[set2])
    self.bit_groups[set1].extend(self.bit_groups[set2])
    self.gate_groups[set2].clear()
    self.bit_groups[set2].clear()