from __future__ import annotations
from collections.abc import Generator
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
Run rescheduler.

        This pass should perform rescheduling to satisfy:

            - All DAGOpNode nodes (except for compiler directives) are placed at start time
              satisfying hardware alignment constraints.
            - The end time of a node does not overlap with the start time of successor nodes.

        Assumptions:

            - Topological order and absolute time order of DAGOpNode are consistent.
            - All bits in either qargs or cargs associated with node synchronously start.
            - Start time of qargs and cargs may different due to I/O latency.

        Based on the configurations above, the rescheduler pass takes the following strategy:

        1. The nodes are processed in the topological order, from the beginning of
            the circuit (i.e. from left to right). For every node (including compiler
            directives), the function ``_push_node_back`` performs steps 2 and 3.
        2. If the start time of the node violates the alignment constraint,
            the start time is increased to satisfy the constraint.
        3. Each immediate successor whose start_time overlaps the node's end_time is
            pushed backwards (towards the end of the wire). Note that at this point
            the shifted successor does not need to satisfy the constraints, but this
            will be taken care of when that successor node itself is processed.
        4. After every node is processed, all misalignment constraints will be resolved,
            and there will be no overlap between the nodes.

        Args:
            dag: DAG circuit to be rescheduled with constraints.

        Raises:
            TranspilerError: If circuit is not scheduled.
        