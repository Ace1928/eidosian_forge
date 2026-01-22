from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag
class UnrollForLoops(TransformationPass):
    """``UnrollForLoops`` transpilation pass unrolls for-loops when possible."""

    def __init__(self, max_target_depth=-1):
        """Things like ``for x in {0, 3, 4} {rx(x) qr[1];}`` will turn into
        ``rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];``.

        .. note::
            The ``UnrollForLoops`` unrolls only one level of block depth. No inner loop will
            be considered by ``max_target_depth``.

        Args:
            max_target_depth (int): Optional. Checks if the unrolled block is over a particular
                subcircuit depth. To disable the check, use ``-1`` (Default).
        """
        super().__init__()
        self.max_target_depth = max_target_depth

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the UnrollForLoops pass on ``dag``.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for forloop_op in dag.op_nodes(ForLoopOp):
            indexset, loop_param, body = forloop_op.op.params
            if 0 < self.max_target_depth < len(indexset) * body.depth():
                continue
            if _body_contains_continue_or_break(body):
                continue
            unrolled_dag = circuit_to_dag(body).copy_empty_like()
            for index_value in indexset:
                bound_body = body.assign_parameters({loop_param: index_value}) if loop_param else body
                unrolled_dag.compose(circuit_to_dag(bound_body), inplace=True)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag)
        return dag