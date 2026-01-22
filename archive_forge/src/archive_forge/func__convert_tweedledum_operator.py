from qiskit.utils.optionals import HAS_TWEEDLEDUM
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
@HAS_TWEEDLEDUM.require_in_call
def _convert_tweedledum_operator(op):
    base_gate = _QISKIT_OPS.get(op.kind())
    if base_gate is None:
        if op.kind() == 'py_operator':
            return op.py_op()
        else:
            raise RuntimeError('Unrecognized operator: %s' % op.kind())
    if op.num_controls() > 0:
        from tweedledum.ir import Qubit
        qubits = op.qubits()
        ctrl_state = ''
        for qubit in qubits[:op.num_controls()]:
            ctrl_state += f'{int(qubit.polarity() == Qubit.Polarity.positive)}'
        return base_gate().control(len(ctrl_state), ctrl_state=ctrl_state[::-1])
    return base_gate()