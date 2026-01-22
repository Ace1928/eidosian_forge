from qiskit.transpiler.basepasses import AnalysisPass
class Collect1qRuns(AnalysisPass):
    """Collect one-qubit subcircuits."""

    def run(self, dag):
        """Run the Collect1qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['run_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set['run_list'] = dag.collect_1q_runs()
        return dag