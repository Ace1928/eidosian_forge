import warnings
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.circuit import Delay, Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
@staticmethod
def _get_node_duration(node: DAGOpNode, dag: DAGCircuit) -> int:
    """A helper method to get duration from node or calibration."""
    indices = [dag.find_bit(qarg).index for qarg in node.qargs]
    if dag.has_calibration_for(node):
        cal_key = (tuple(indices), tuple((float(p) for p in node.op.params)))
        duration = dag.calibrations[node.op.name][cal_key].duration
        node.op = node.op.to_mutable()
        node.op.duration = duration
    else:
        duration = node.op.duration
    if isinstance(duration, ParameterExpression):
        raise TranspilerError(f'Parameterized duration ({duration}) of {node.op.name} on qubits {indices} is not bounded.')
    if duration is None:
        raise TranspilerError(f'Duration of {node.op.name} on qubits {indices} is not found.')
    return duration