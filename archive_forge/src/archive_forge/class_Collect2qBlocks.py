from collections import defaultdict
from qiskit.transpiler.basepasses import AnalysisPass
class Collect2qBlocks(AnalysisPass):
    """Collect two-qubit subcircuits."""

    def run(self, dag):
        """Run the Collect2qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['block_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set['commutation_set'] = defaultdict(list)
        self.property_set['block_list'] = dag.collect_2q_runs()
        return dag