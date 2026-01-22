from qiskit.transpiler.basepasses import AnalysisPass
class Depth(AnalysisPass):
    """Calculate the depth of a DAG circuit."""

    def __init__(self, *, recurse=False):
        """
        Args:
            recurse: whether to allow recursion into control flow.  If this is ``False`` (default),
                the pass will throw an error when control flow is present, to avoid returning a
                number with little meaning.
        """
        super().__init__()
        self.recurse = recurse

    def run(self, dag):
        """Run the Depth pass on `dag`."""
        self.property_set['depth'] = dag.depth(recurse=self.recurse)