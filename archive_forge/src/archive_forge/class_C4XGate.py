from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=4, cached_states=(15,))
class C4XGate(SingletonControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1], with the use
    of the relative phase version of c3x, the rc3x [2].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        [2] Maslov, 2015. https://arxiv.org/abs/1508.03273
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create a new 4-qubit controlled X gate."""
        if unit is None:
            unit = 'dt'
        super().__init__('mcx', 5, [], num_ctrl_qubits=4, label=label, ctrl_state=ctrl_state, base_gate=XGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=4)

    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(pi/8) a,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
        }
        gate c4x a,b,c,d,e
        {
            h e; cu1(pi/2) d,e; h e;
            rc3x a,b,c,d;
            h e; cu1(-pi/2) d,e; h e;
            rc3x a,b,c,d;
            c3sqrtx a,b,c,e;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        from .h import HGate
        q = QuantumRegister(5, name='q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[4]], []), (CU1Gate(numpy.pi / 2), [q[3], q[4]], []), (HGate(), [q[4]], []), (RC3XGate(), [q[0], q[1], q[2], q[3]], []), (HGate(), [q[4]], []), (CU1Gate(-numpy.pi / 2), [q[3], q[4]], []), (HGate(), [q[4]], []), (RC3XGate().inverse(), [q[0], q[1], q[2], q[3]], []), (C3SXGate(), [q[0], q[1], q[2], q[4]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = self.ctrl_state << num_ctrl_qubits | ctrl_state
            gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 4, label=label, ctrl_state=new_ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Invert this gate. The C4X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C4XGate: inverse gate (self-inverse).
        """
        return C4XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C4XGate) and self.ctrl_state == other.ctrl_state