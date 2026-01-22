from qiskit.circuit import Measure
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler
class ASAPScheduleAnalysis(BaseScheduler):
    """ASAP Scheduling pass, which schedules the start time of instructions as early as possible.

    See the :ref:`scheduling_stage` section in the :mod:`qiskit.transpiler`
    module documentation for the detailed behavior of the control flow
    operation, i.e. ``c_if``.
    """

    def run(self, dag):
        """Run the ASAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('ASAP schedule runs on physical circuits only')
        conditional_latency = self.property_set.get('conditional_latency', 0)
        clbit_write_latency = self.property_set.get('clbit_write_latency', 0)
        node_start_time = {}
        idle_after = {q: 0 for q in dag.qubits + dag.clbits}
        for node in dag.topological_op_nodes():
            op_duration = self._get_node_duration(node, dag)
            if isinstance(node.op, self.CONDITIONAL_SUPPORTED):
                t0q = max((idle_after[q] for q in node.qargs))
                if node.op.condition_bits:
                    t0c = max((idle_after[bit] for bit in node.op.condition_bits))
                    if t0q > t0c:
                        t0c = max(t0q - conditional_latency, t0c)
                    t1c = t0c + conditional_latency
                    for bit in node.op.condition_bits:
                        idle_after[bit] = t1c
                    t0 = max(t0q, t1c)
                else:
                    t0 = t0q
                t1 = t0 + op_duration
            else:
                if node.op.condition_bits:
                    raise TranspilerError(f'Conditional instruction {node.op.name} is not supported in ASAP scheduler.')
                if isinstance(node.op, Measure):
                    t0q = max((idle_after[q] for q in node.qargs))
                    t0c = max((idle_after[c] for c in node.cargs))
                    t0 = max(t0q, t0c - clbit_write_latency)
                    t1 = t0 + op_duration
                    for clbit in node.cargs:
                        idle_after[clbit] = t1
                else:
                    t0 = max((idle_after[bit] for bit in node.qargs + node.cargs))
                    t1 = t0 + op_duration
            for bit in node.qargs:
                idle_after[bit] = t1
            node_start_time[node] = t0
        self.property_set['node_start_time'] = node_start_time