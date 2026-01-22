from collections import defaultdict
import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOutNode
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit import ControlFlowOp

        This is similar to transpiler/passes/utils/control_flow.py except that the
        commutation analysis is redone for the control flow blocks.
        