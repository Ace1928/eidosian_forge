from typing import (
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types, control_values as cv
from cirq.type_workarounds import NotImplementedType
from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate
imports.
@value.value_equality
class XPowGate(eigen_gate.EigenGate):
    """A gate that rotates around the X axis of the Bloch sphere.

    The unitary matrix of `cirq.XPowGate(exponent=t, global_shift=s)` is:
    $$
    e^{i \\pi t (s + 1/2)}
    \\begin{bmatrix}
      \\cos(\\pi t /2) & -i \\sin(\\pi t /2) \\\\
      -i \\sin(\\pi t /2) & \\cos(\\pi t /2)
    \\end{bmatrix}
    $$

    Note in particular that this gate has a global phase factor of
    $e^{i \\pi t / 2}$ vs the traditionally defined rotation matrices
    about the Pauli X axis. See `cirq.Rx` for rotations without the global
    phase. The global phase factor can be adjusted by using the `global_shift`
    parameter when initializing.

    `cirq.X`, the Pauli X gate, is an instance of this gate at `exponent=1`.
    """
    _eigencomponents: Dict[int, List[Tuple[float, np.ndarray]]] = {}

    def __init__(self, *, exponent: value.TParamVal=1.0, global_shift: float=0.0, dimension: int=2):
        """Initialize an XPowGate.

        Args:
            exponent: The t in gate**t. Determines how much the eigenvalues of
                the gate are phased by. For example, eigenvectors phased by -1
                when `gate**1` is applied will gain a relative phase of
                e^{i pi exponent} when `gate**exponent` is applied (relative to
                eigenvectors unaffected by `gate**1`).
            global_shift: Offsets the eigenvalues of the gate at exponent=1.
                In effect, this controls a global phase factor on the gate's
                unitary matrix. The factor for global_shift=s is:

                    exp(i * pi * s * t)

                For example, `cirq.X**t` uses a `global_shift` of 0 but
                `cirq.rx(t)` uses a `global_shift` of -0.5, which is why
                `cirq.unitary(cirq.rx(pi))` equals -iX instead of X.
            dimension: Qudit dimension of this gate. For qu*b*its (the default),
                this is set to 2.

        Raises:
            ValueError: If the supplied exponent is a complex number with an
                imaginary component.
        """
        super().__init__(exponent=exponent, global_shift=global_shift)
        self._dimension = dimension

    @property
    def dimension(self) -> value.TParamVal:
        return self._dimension

    def _num_qubits_(self) -> int:
        return 1

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1 or self._dimension != 2:
            return NotImplemented
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.available_buffer *= p
        return args.available_buffer

    def in_su2(self) -> 'Rx':
        """Returns an equal-up-global-phase gate from the group SU2."""
        return Rx(rads=self._exponent * _pi(self._exponent))

    def with_canonical_global_phase(self) -> 'XPowGate':
        """Returns an equal-up-global-phase standardized form of the gate."""
        return XPowGate(exponent=self._exponent, dimension=self._dimension)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self._dimension,)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        if self._dimension not in XPowGate._eigencomponents:
            components = []
            root = 1j ** (4 / self._dimension)
            for i in range(self._dimension):
                half_turns = i * 2 / self._dimension
                v = np.array([root ** (i * j) / self._dimension for j in range(self._dimension)])
                m = np.array([np.roll(v, j) for j in range(self._dimension)])
                components.append((half_turns, m))
            XPowGate._eigencomponents[self._dimension] = components
        return XPowGate._eigencomponents[self._dimension]

    def _with_exponent(self, exponent: 'cirq.TParamVal') -> 'cirq.XPowGate':
        return XPowGate(exponent=exponent, global_shift=self._global_shift, dimension=self._dimension)

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate
        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return SingleQubitCliffordGate.X_sqrt.on(*qubits)
        if self.exponent % 2 == 1:
            return SingleQubitCliffordGate.X.on(*qubits)
        if self.exponent % 2 == 1.5:
            return SingleQubitCliffordGate.X_nsqrt.on(*qubits)
        return NotImplemented

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> raw_types.Gate:
        """Returns a controlled `XPowGate`, using a `CXPowGate` where possible.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate`. This method overrides this behavior
        to return a `CXPowGate` or a `ControlledGate` of a `CXPowGate`, when
        this is possible.

        The conditions for the override to occur are:

        * The `global_shift` of the `XPowGate` is 0.
        * The `control_values` and `control_qid_shape` are compatible with
            the `CXPowGate`:
            * The last value of `control_qid_shape` is a qubit.
            * The last value of `control_values` corresponds to the
                control being satisfied if that last qubit is 1 and
                not satisfied if the last qubit is 0.

        If these conditions are met, then the returned object is a `CXPowGate`
        or, in the case that there is more than one controlled qudit, a
        `ControlledGate` with the `Gate` being a `CXPowGate`. In the
        latter case the `ControlledGate` is controlled by one less qudit
        than specified in `control_values` and `control_qid_shape` (since
        one of these, the last qubit, is used as the control for the
        `CXPowGate`).

        If the above conditions are not met, a `ControlledGate` of this
        gate will be returned.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.ControlledGate` (or `cirq.CXPowGate` if possible) representing
                `self` controlled by the given control values and qubits.
        """
        if control_values and (not isinstance(control_values, cv.AbstractControlValues)):
            control_values = cv.ProductOfSums(tuple(((val,) if isinstance(val, int) else tuple(sorted(val)) for val in control_values)))
        result = super().controlled(num_controls, control_values, control_qid_shape)
        if self._global_shift == 0 and isinstance(result, controlled_gate.ControlledGate) and isinstance(result.control_values, cv.ProductOfSums) and (result.control_values[-1] == (1,)) and (result.control_qid_shape[-1] == 2):
            return cirq.CXPowGate(exponent=self._exponent, global_shift=self._global_shift).controlled(result.num_controls() - 1, result.control_values[:-1], result.control_qid_shape[:-1])
        return result

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self) or self._dimension != 2:
            return NotImplemented
        phase = 1j ** (2 * self._exponent * (self._global_shift + 0.5))
        angle = np.pi * self._exponent / 2
        return value.LinearDict({'I': phase * np.cos(angle), 'X': -1j * phase * np.sin(angle)})

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=('X',), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._global_shift == 0:
            if self._exponent == 1:
                return args.format('x {0};\n', qubits[0])
            elif self._exponent == 0.5:
                return args.format('sx {0};\n', qubits[0])
            elif self._exponent == -0.5:
                return args.format('sxdg {0};\n', qubits[0])
        return args.format('rx({0:half_turns}) {1};\n', self._exponent, qubits[0])

    @property
    def phase_exponent(self):
        return 0.0

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(exponent=self._exponent, phase_exponent=phase_turns * 2)

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_() or self._dimension != 2:
            return None
        return self.exponent % 0.5 == 0

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'X'
            return f'X**{self._exponent}'
        return f'XPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0 and self._dimension == 2:
            if self._exponent == 1:
                return 'cirq.X'
            return f'(cirq.X**{proper_repr(self._exponent)})'
        args = []
        if self._exponent != 1:
            args.append(f'exponent={proper_repr(self._exponent)}')
        if self._global_shift != 0:
            args.append(f'global_shift={self._global_shift}')
        if self._dimension != 2:
            args.append(f'dimension={self._dimension}')
        all_args = ', '.join(args)
        return f'cirq.XPowGate({all_args})'

    def _json_dict_(self) -> Dict[str, Any]:
        d = protocols.obj_to_dict_helper(self, ['exponent', 'global_shift'])
        if self.dimension != 2:
            d['dimension'] = self.dimension
        return d

    def _value_equality_values_(self):
        return (*super()._value_equality_values_(), self._dimension)

    def _value_equality_approximate_values_(self):
        return (*super()._value_equality_approximate_values_(), self._dimension)