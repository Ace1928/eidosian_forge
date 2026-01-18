import functools
from typing import Callable
from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
def trivial_recurse(method):
    """Decorator that causes :class:`.BasePass.run` to iterate over all control-flow nodes,
    replacing their operations with a new :class:`.ControlFlowOp` whose blocks have all had
    :class`.BasePass.run` called on them.

    This is only suitable for simple run calls that store no state between calls, do not need
    circuit-specific information feeding into them (such as via a :class:`.PropertySet`), and will
    safely do nothing to control-flow operations that are in the DAG.

    If slightly finer control is needed on when the control-flow operations are modified, one can
    use :func:`map_blocks` as::

        if isinstance(node.op, ControlFlowOp):
            node.op = map_blocks(self.run, node.op)

    from with :meth:`.BasePass.run`."""

    @functools.wraps(method)
    def out(self, dag):

        def bound_wrapped_method(dag):
            return out(self, dag)
        for node in dag.op_nodes(ControlFlowOp):
            node.op = map_blocks(bound_wrapped_method, node.op)
        return method(self, dag)
    return out