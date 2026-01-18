from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.reset import Reset
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
Run the pass on a dag.