from qiskit.circuit.barrier import Barrier
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from .merge_adjacent_barriers import MergeAdjacentBarriers
Run the BarrierBeforeFinalMeasurements pass on `dag`.