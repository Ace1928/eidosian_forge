from itertools import groupby
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.quaternion import Quaternion
from qiskit._accelerate.optimize_1q_gates import compose_u3_rust
def _split_runs_on_parameters(runs):
    """Finds runs containing parameterized gates and splits them into sequential
    runs excluding the parameterized gates.
    """
    out = []
    for run in runs:
        groups = groupby(run, lambda x: x.op.is_parameterized() and x.op.name in ('u3', 'u'))
        for group_is_parameterized, gates in groups:
            if not group_is_parameterized:
                out.append(list(gates))
    return out