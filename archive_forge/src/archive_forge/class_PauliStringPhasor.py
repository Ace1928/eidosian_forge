from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
@value.value_equality(approximate=True)
class PauliStringPhasor(gate_operation.GateOperation):
    """An operation that phases the eigenstates of a Pauli string.

    This class takes `PauliString`, which is a sequence of non-identity
    Pauli operators, potentially with a $\\pm 1$ valued coefficient,
    acting on qubits.

    The -1 eigenstates of the Pauli string will have their amplitude multiplied
    by e^(i pi exponent_neg) while +1 eigenstates of the Pauli string will have
    their amplitude multiplied by e^(i pi exponent_pos).

    The class also takes a list of qubits, which can be a superset of those
    acted on by the provided `PauliString`.  Those extra qubits are assumed to be
    acted upon via identity.
    """

    def __init__(self, pauli_string: ps.PauliString, qubits: Optional[Sequence['cirq.Qid']]=None, *, exponent_neg: Union[int, float, sympy.Expr]=1, exponent_pos: Union[int, float, sympy.Expr]=0) -> None:
        """Initializes the operation.

        Args:
            pauli_string: The PauliString defining the positive and negative
                eigenspaces that will be independently phased.
            qubits: The qubits upon which the PauliStringPhasor acts. This
                must be a superset of the qubits of `pauli_string`.
                If None, it will use the qubits from `pauli_string`
                The `pauli_string` contains only the non-identity component
                of the phasor, while the qubits supplied here and not in
                `pauli_string` are acted upon by identity. The order of
                these qubits must match the order in `pauli_string`.
            exponent_neg: How much to phase vectors in the negative eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).
            exponent_pos: How much to phase vectors in the positive eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).

        Raises:
            ValueError: If coefficient is not 1 or -1 or the qubits of
                `pauli_string` are not a subset of `qubits`.
        """
        if qubits is not None:
            it = iter(qubits)
            if any((not any((q0 == q1 for q1 in it)) for q0 in pauli_string.qubits)):
                raise ValueError(f"PauliStringPhasor's pauli string qubits ({pauli_string.qubits}) are not an ordered subset of the explicit qubits ({qubits}).")
        else:
            qubits = pauli_string.qubits
        gate = PauliStringPhasorGate(pauli_string.dense(qubits), exponent_neg=exponent_neg, exponent_pos=exponent_pos)
        super().__init__(gate, qubits)
        self._pauli_string = gate.dense_pauli_string.on(*self.qubits)

    @property
    def gate(self) -> 'cirq.PauliStringPhasorGate':
        """The gate applied by the operation."""
        return cast(PauliStringPhasorGate, self._gate)

    @property
    def exponent_neg(self) -> Union[int, float, sympy.Expr]:
        """The negative exponent."""
        return self.gate.exponent_neg

    @property
    def exponent_pos(self) -> Union[int, float, sympy.Expr]:
        """The positive exponent."""
        return self.gate.exponent_pos

    @property
    def pauli_string(self) -> 'cirq.PauliString':
        """The underlying pauli string."""
        return self._pauli_string

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Expr]:
        """The relative exponent between negative and positive exponents."""
        return self.gate.exponent_relative

    def _value_equality_values_(self):
        return (self.pauli_string, self.qubits, self.exponent_neg, self.exponent_pos)

    def equal_up_to_global_phase(self, other: 'PauliStringPhasor') -> bool:
        """Checks equality of two PauliStringPhasors, up to global phase."""
        if isinstance(other, PauliStringPhasor):
            return self.exponent_relative == other.exponent_relative and self.pauli_string == other.pauli_string and (self.qubits == other.qubits)
        return False

    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]) -> 'PauliStringPhasor':
        """Maps the qubits inside the PauliStringPhasor.

        Args:
            qubit_map: A map from the qubits in the phasor to new qubits.

        Returns:
            A new PauliStringPhasor with remapped qubits.

        Raises:
            ValueError: If the map does not contain an entry for all
                the qubits in the phasor.
        """
        if not set(self.qubits) <= qubit_map.keys():
            raise ValueError(f'qubit_map must have a key for every qubit in the phasors qubits. keys: {qubit_map.keys()} phasor qubits: {self.qubits}')
        return PauliStringPhasor(pauli_string=self.pauli_string.map_qubits(qubit_map), qubits=[qubit_map[q] for q in self.qubits], exponent_neg=self.exponent_neg, exponent_pos=self.exponent_pos)

    def can_merge_with(self, op: 'PauliStringPhasor') -> bool:
        """Checks whether the underlying PauliStrings can be merged."""
        return self.pauli_string.equal_up_to_coefficient(op.pauli_string) and self.qubits == op.qubits

    def merged_with(self, op: 'PauliStringPhasor') -> 'PauliStringPhasor':
        """Merges two PauliStringPhasors."""
        if not self.can_merge_with(op):
            raise ValueError(f'Cannot merge operations: {self}, {op}')
        pp = self.exponent_pos + op.exponent_pos
        pn = self.exponent_neg + op.exponent_neg
        return PauliStringPhasor(self.pauli_string, qubits=self.qubits, exponent_pos=pp, exponent_neg=pn)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        qubits = self.qubits if args.known_qubits is None else args.known_qubits

        def sym(qubit):
            if qubit in self.pauli_string:
                return f'[{self.pauli_string[qubit]}]'
            return '[I]'
        syms = tuple((sym(qubit) for qubit in qubits))
        return protocols.CircuitDiagramInfo(wire_symbols=syms, exponent=self.exponent_relative)

    def pass_operations_over(self, ops: Iterable[raw_types.Operation], after_to_before: bool=False) -> 'PauliStringPhasor':
        """Determines how the Pauli phasor changes when conjugated by Cliffords.

        The output and input pauli phasors are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_PHASOR───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_PHASOR───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
        new_pauli_string = self.pauli_string.pass_operations_over(ops, after_to_before)
        pp = self.exponent_pos
        pn = self.exponent_neg
        return PauliStringPhasor(new_pauli_string, exponent_pos=pp, exponent_neg=pn)

    def __repr__(self) -> str:
        return f'cirq.PauliStringPhasor({self.pauli_string!r}, qubits={self.qubits!r}, exponent_neg={proper_repr(self.exponent_neg)}, exponent_pos={proper_repr(self.exponent_pos)})'

    def __str__(self) -> str:
        if self.exponent_pos == -self.exponent_neg:
            sign = '-' if self.exponent_pos < 0 else ''
            exponent = str(abs(self.exponent_pos))
            return f'exp({sign}iπ{exponent}*{self.pauli_string})'
        return f'({self.pauli_string})**{self.exponent_relative}'

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['pauli_string', 'qubits', 'exponent_neg', 'exponent_pos'])

    @classmethod
    def _from_json_dict_(cls, pauli_string, exponent_neg, exponent_pos, **kwargs):
        qubits = kwargs['qubits'] if 'qubits' in kwargs else None
        return PauliStringPhasor(pauli_string=pauli_string, qubits=qubits, exponent_neg=exponent_neg, exponent_pos=exponent_pos)