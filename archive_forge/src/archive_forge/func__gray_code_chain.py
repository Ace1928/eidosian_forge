import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
def _gray_code_chain(q, num_ctrl_qubits, gate):
    """Apply the gate to the last qubit in the register ``q``, controlled on all
    preceding qubits. This function uses the gray code to propagate down to the last qubit.

    Ported and adapted from Aqua (github.com/Qiskit/qiskit-aqua),
    commit 769ca8d, file qiskit/aqua/circuits/gates/multi_control_u1_gate.py.
    """
    from .x import CXGate
    rule = []
    q_controls, q_target = (q[:num_ctrl_qubits], q[num_ctrl_qubits])
    gray_code = _generate_gray_code(num_ctrl_qubits)
    last_pattern = None
    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        lm_pos = list(pattern).index('1')
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                rule.append((CXGate(), [q_controls[pos], q_controls[lm_pos]], []))
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    rule.append((CXGate(), [q_controls[idx], q_controls[lm_pos]], []))
        if pattern.count('1') % 2 == 0:
            rule.append((gate.inverse(), [q_controls[lm_pos], q_target], []))
        else:
            rule.append((gate, [q_controls[lm_pos], q_target], []))
        last_pattern = pattern
    return rule