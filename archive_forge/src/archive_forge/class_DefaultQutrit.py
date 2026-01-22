import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
class DefaultQutrit(QutritDevice):
    """Default qutrit device for PennyLane.

    .. warning::

        The API of ``DefaultQutrit`` will be updated soon to follow a new device interface described
        in :class:`pennylane.devices.Device`.

        This change will not alter device behaviour for most workflows, but may have implications for
        plugin developers and users who directly interact with device methods. Please consult
        :class:`pennylane.devices.Device` and the implementation in
        :class:`pennylane.devices.DefaultQubit` for more information on what the new
        interface will look like and be prepared to make updates in a coming release. If you have any
        feedback on these changes, please create an
        `issue <https://github.com/PennyLaneAI/pennylane/issues>`_ or post in our
        `discussion forum <https://discuss.pennylane.ai/>`_.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """
    name = 'Default qutrit PennyLane plugin'
    short_name = 'default.qutrit'
    pennylane_requires = __version__
    version = __version__
    author = 'Mudit Pandey, UBC Quantum Software and Algorithms Research Group, and Xanadu'
    operations = {'Identity', 'QutritUnitary', 'ControlledQutritUnitary', 'TShift', 'Adjoint(TShift)', 'TClock', 'Adjoint(TClock)', 'TAdd', 'Adjoint(TAdd)', 'TSWAP', 'THadamard', 'Adjoint(THadamard)', 'TRX', 'TRY', 'TRZ', 'QutritBasisState'}
    observables = {'THermitian', 'GellMann', 'Identity'}
    _reshape = staticmethod(qml.math.reshape)
    _flatten = staticmethod(qml.math.flatten)
    _transpose = staticmethod(qml.math.transpose)
    _dot = staticmethod(qml.math.dot)
    _stack = staticmethod(qml.math.stack)
    _conj = staticmethod(qml.math.conj)
    _roll = staticmethod(qml.math.roll)
    _cast = staticmethod(qml.math.cast)
    _tensordot = staticmethod(qml.math.tensordot)
    _real = staticmethod(qml.math.real)
    _imag = staticmethod(qml.math.imag)

    @staticmethod
    def _reduce_sum(array, axes):
        return qml.math.sum(array, tuple(axes))

    @staticmethod
    def _asarray(array, dtype=None):
        if not hasattr(array, '__len__'):
            return np.asarray(array, dtype=dtype)
        res = qml.math.cast(array, dtype=dtype)
        return res

    def __init__(self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state
        self._apply_ops = {'TShift': self._apply_tshift, 'TClock': self._apply_tclock, 'TAdd': self._apply_tadd, 'TSWAP': self._apply_tswap}

    @functools.lru_cache()
    def map_wires(self, wires):
        try:
            mapped_wires = [self.wire_map[w] for w in wires]
        except KeyError as e:
            raise WireError(f'Did not find some of the wires {wires.labels} on device with wires {self.wires.labels}.') from e
        return mapped_wires

    def define_wire_map(self, wires):
        consecutive_wires = range(self.num_wires)
        wire_map = zip(wires, consecutive_wires)
        return dict(wire_map)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []
        for i, operation in enumerate(operations):
            if i > 0 and isinstance(operation, QutritBasisState):
                raise DeviceError(f'Operation {operation.name} cannot be used after other operations have already been applied on a {self.short_name} device.')
            if isinstance(operation, QutritBasisState):
                self._apply_basis_state(operation.parameters[0], operation.wires)
            else:
                self._state = self._apply_operation(self._state, operation)
        self._pre_rotated_state = self._state
        for operation in rotations:
            self._state = self._apply_operation(self._state, operation)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s, 1s and 2s.
            wires (Wires): wires that the provided computational state should be initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        device_wires = self.map_wires(wires)
        n_basis_state = len(state)
        if not set(state.tolist()).issubset({0, 1, 2}):
            raise ValueError('QutritBasisState parameter must consist of 0, 1 or 2 integers.')
        if n_basis_state != len(device_wires):
            raise ValueError('QutritBasisState parameter and wires must be of equal length.')
        basis_states = 3 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))
        self._state = self._create_basis_state(num)

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        if operation.name == 'Identity':
            return state
        wires = operation.wires
        if operation.name in self._apply_ops:
            axes = self.wires.indices(wires)
            return self._apply_ops[operation.name](state, axes)
        elif isinstance(operation, qml.ops.Adjoint) and operation.base.name in self._apply_ops:
            axes = self.wires.indices(wires)
            return self._apply_ops[operation.base.name](state, axes, inverse=True)
        matrix = self._asarray(self._get_unitary_matrix(operation), dtype=self.C_DTYPE)
        return self._apply_unitary(state, matrix, wires)

    def _apply_tshift(self, state, axes, inverse=False):
        """Applies a ternary Shift gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|2 \rangle` is transformed to
        :math:`|0 \rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        shift = -1 if inverse else 1
        return self._roll(state, shift, axes[0])

    def _apply_tclock(self, state, axes, inverse=False):
        """Applies a ternary Clock gate by adding appropriate phases to the 1 and 2 indices
        along the axis specified in ``axes``

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        partial_state = self._apply_phase(state, axes, 1, OMEGA, inverse)
        return self._apply_phase(partial_state, axes, 2, OMEGA ** 2, inverse)

    def _apply_tadd(self, state, axes, inverse=False):
        """Applies a controlled ternary add gate by slicing along the first axis specified in ``axes`` and
        applying a TShift transformation along the second axis. The ternary add gate acts on the computational
        basis states like :math:`	ext{TAdd}\x0bert i, j\rangle \rightarrow \x0bert i, i+j \rangle`, where addition
        is taken modulo 3.

        By slicing along the first axis, we are able to select all of the amplitudes with corresponding
        :math:`|1\rangle` and :math:`|2\rangle` for the control qutrit. This means we just need to apply
        a :class:`~.TShift` gate when slicing along index 1, and a :class:`~.TShift` adjoint gate when
        slicing along index 2

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        slices = [_get_slice(i, axes[0], self.num_wires) for i in range(3)]
        target_axes = [axes[1] - 1] if axes[1] > axes[0] else [axes[1]]
        state_1 = self._apply_tshift(state[slices[1]], axes=target_axes, inverse=inverse)
        state_2 = self._apply_tshift(state[slices[2]], axes=target_axes, inverse=not inverse)
        return self._stack([state[slices[0]], state_1, state_2], axis=axes[0])

    def _apply_tswap(self, state, axes, **kwargs):
        """Applies a ternary SWAP gate by performing a partial transposition along the
        specified axes. The ternary SWAP gate acts on the computational basis states like
        :math:`\x0bert i, j\rangle \rightarrow \x0bert j, i \rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        all_axes = list(range(len(state.shape)))
        all_axes[axes[0]] = axes[1]
        all_axes[axes[1]] = axes[0]
        return self._transpose(state, all_axes)

    def _apply_phase(self, state, axes, index, phase, inverse=False):
        """Applies a phase onto the specified index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            index (int): target index of axis to apply phase to
            phase (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
        num_wires = len(state.shape)
        slices = [_get_slice(i, axes[0], num_wires) for i in range(3)]
        phase = self._conj(phase) if inverse else phase
        state_slices = [self._const_mul(phase if i == index else 1, state[slices[i]]) for i in range(3)]
        return self._stack(state_slices, axis=axes[0])

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis.
        """
        return unitary.matrix()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(model='qutrit', supports_inverse_operations=True, supports_analytic_computation=True, returns_state=True, passthru_devices={'autograd': 'default.qutrit', 'tf': 'default.qutrit', 'torch': 'default.qutrit', 'jax': 'default.qutrit'})
        return capabilities

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[3]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(3 ** self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [3] * self.num_wires)

    @property
    def state(self):
        return self._flatten(self._pre_rotated_state)

    def density_matrix(self, wires):
        """Returns the reduced density matrix of a given set of wires.

        Args:
            wires (Wires): wires of the reduced system.

        Returns:
            array[complex]: complex tensor of shape ``(3 ** len(wires), 3 ** len(wires))``
            representing the reduced density matrix.
        """
        dim = self.num_wires
        state = self._pre_rotated_state
        if wires == self.wires:
            density_matrix = self._tensordot(state, self._conj(state), axes=0)
            density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))
            return density_matrix
        complete_system = list(range(0, dim))
        traced_system = [x for x in complete_system if x not in wires.labels]
        density_matrix = self._tensordot(state, self._conj(state), axes=(traced_system, traced_system))
        density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))
        return density_matrix

    def _apply_unitary(self, state, mat, wires):
        """Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state
        """
        device_wires = self.map_wires(wires)
        mat = self._cast(self._reshape(mat, [3] * len(device_wires) * 2), dtype=self.C_DTYPE)
        axes = (list(range(len(device_wires), 2 * len(device_wires))), device_wires)
        tdot = self._tensordot(mat, state, axes=axes)
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wires]
        perm = list(device_wires) + unused_idxs
        inv_perm = np.argsort(perm)
        return self._transpose(tdot, inv_perm)

    def reset(self):
        """Reset the device"""
        super().reset()
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None
        flat_state = self._flatten(self._state)
        real_state = self._real(flat_state)
        imag_state = self._imag(flat_state)
        prob = self.marginal_prob(real_state ** 2 + imag_state ** 2, wires)
        return prob