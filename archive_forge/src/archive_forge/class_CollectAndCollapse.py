from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockCollapser
from qiskit.transpiler.passes.utils import control_flow
class CollectAndCollapse(TransformationPass):
    """A general transpiler pass to collect and to consolidate blocks of nodes
    in a circuit.

    This transpiler pass depends on two functions: the collection function and the
    collapsing function. The collection function ``collect_function`` takes a DAG
    and returns a list of blocks. The collapsing function ``collapse_function``
    takes a DAG and a list of blocks, consolidates each block, and returns the modified
    DAG.

    The input and the output DAGs are of type :class:`~qiskit.dagcircuit.DAGCircuit`,
    however when exploiting commutativity analysis to collect blocks, the
    :class:`~qiskit.dagcircuit.DAGDependency` representation is used internally.
    To support this, the ``collect_function`` and ``collapse_function`` should work
    with both types of DAGs and DAG nodes.

    Other collection and consolidation transpiler passes, for instance
    :class:`~.CollectLinearFunctions`, may derive from this pass, fixing
    ``collect_function`` and ``collapse_function`` to specific functions.
    """

    def __init__(self, collect_function, collapse_function, do_commutative_analysis=False):
        """
        Args:
            collect_function (callable): a function that takes a DAG and returns a list
                of "collected" blocks of nodes
            collapse_function (callable): a function that takes a DAG and a list of
                "collected" blocks, and consolidates each block.
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
        """
        self.collect_function = collect_function
        self.collapse_function = collapse_function
        self.do_commutative_analysis = do_commutative_analysis
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the CollectLinearFunctions pass on `dag`.
        Args:
            dag (DAGCircuit): the DAG to be optimized.
        Returns:
            DAGCircuit: the optimized DAG.
        """
        if self.do_commutative_analysis:
            dag = dag_to_dagdependency(dag)
        blocks = self.collect_function(dag)
        self.collapse_function(dag, blocks)
        if self.do_commutative_analysis:
            dag = dagdependency_to_dag(dag)
        return dag