import functools
import itertools
from collections import defaultdict
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
import pennylane.math as qnp
from pennylane import (
from pennylane.measurements import CountsMP, MutualInfoMP, SampleMP, StateMP, VnEntropyMP, PurityMP
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.wires import Wires
from .._version import __version__
class DefaultMixed(QubitDevice):
    """Default qubit device for performing mixed-state computations in PennyLane.

    .. warning::

        The API of ``DefaultMixed`` will be updated soon to follow a new device interface described
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
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (None, int): Number of times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that
            outputs are computed exactly.
        readout_prob (None, int, float): Probability for adding readout error to the measurement
            outcomes of observables. Defaults to ``None`` if not specified, which means that the outcomes are
            without any readout error.
    """
    name = 'Default mixed-state qubit PennyLane plugin'
    short_name = 'default.mixed'
    pennylane_requires = __version__
    version = __version__
    author = 'Xanadu Inc.'
    operations = {'Identity', 'Snapshot', 'BasisState', 'QubitStateVector', 'StatePrep', 'QubitDensityMatrix', 'QubitUnitary', 'ControlledQubitUnitary', 'BlockEncode', 'MultiControlledX', 'DiagonalQubitUnitary', 'SpecialUnitary', 'PauliX', 'PauliY', 'PauliZ', 'MultiRZ', 'Hadamard', 'S', 'T', 'SX', 'CNOT', 'SWAP', 'ISWAP', 'CSWAP', 'Toffoli', 'CCZ', 'CY', 'CZ', 'CH', 'PhaseShift', 'PCPhase', 'ControlledPhaseShift', 'CPhaseShift00', 'CPhaseShift01', 'CPhaseShift10', 'RX', 'RY', 'RZ', 'Rot', 'CRX', 'CRY', 'CRZ', 'CRot', 'AmplitudeDamping', 'GeneralizedAmplitudeDamping', 'PhaseDamping', 'DepolarizingChannel', 'BitFlip', 'PhaseFlip', 'PauliError', 'ResetError', 'QubitChannel', 'SingleExcitation', 'SingleExcitationPlus', 'SingleExcitationMinus', 'DoubleExcitation', 'DoubleExcitationPlus', 'DoubleExcitationMinus', 'QubitCarry', 'QubitSum', 'OrbitalRotation', 'FermionicSWAP', 'QFT', 'ThermalRelaxationError', 'ECR', 'ParametrizedEvolution', 'GlobalPhase'}
    _reshape = staticmethod(qnp.reshape)
    _flatten = staticmethod(qnp.flatten)
    _transpose = staticmethod(qnp.transpose)
    _gather = staticmethod(lambda *args, axis=0, **kwargs: qnp.gather(*args, **kwargs))
    _dot = staticmethod(qnp.dot)
    measurement_map = defaultdict(lambda: '')
    measurement_map[PurityMP] = 'purity'

    @staticmethod
    def _reduce_sum(array, axes):
        return qnp.sum(array, tuple(axes))

    @staticmethod
    def _asarray(array, dtype=None):
        if not hasattr(array, '__len__'):
            return np.asarray(array, dtype=dtype)
        res = qnp.cast(array, dtype=dtype)
        return res

    def __init__(self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None, readout_prob=None):
        if isinstance(wires, int) and wires > 23:
            raise ValueError('This device does not currently support computations on more than 23 wires')
        self.readout_err = readout_prob
        if self.readout_err is not None:
            if not isinstance(self.readout_err, float) and (not isinstance(self.readout_err, int)):
                raise TypeError('The readout error probability should be an integer or a floating-point number in [0,1].')
            if self.readout_err < 0 or self.readout_err > 1:
                raise ValueError('The readout error probability should be in the range [0,1].')
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state
        self.measured_wires = []
        'List: during execution, stores the list of wires on which measurements are acted for\n        applying the readout error to them when readout_prob is non-zero.'

    def _create_basis_state(self, index):
        """Return the density matrix representing a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state.

        Returns:
            array[complex]: complex array of shape ``[2] * (2 * num_wires)``
            representing the density matrix of the basis state.
        """
        rho = qnp.zeros((2 ** self.num_wires, 2 ** self.num_wires), dtype=self.C_DTYPE)
        rho[index, index] = 1
        return qnp.reshape(rho, [2] * (2 * self.num_wires))

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(returns_state=True, passthru_devices={'autograd': 'default.mixed', 'tf': 'default.mixed', 'torch': 'default.mixed', 'jax': 'default.mixed'})
        return capabilities

    @property
    def state(self):
        """Returns the state density matrix of the circuit prior to measurement"""
        dim = 2 ** self.num_wires
        return qnp.reshape(self._pre_rotated_state, (dim, dim))

    def density_matrix(self, wires):
        """Returns the reduced density matrix over the given wires.

        Args:
            wires (Wires): wires of the reduced system

        Returns:
            array[complex]: complex array of shape ``(2 ** len(wires), 2 ** len(wires))``
            representing the reduced density matrix of the state prior to measurement.
        """
        state = getattr(self, 'state', None)
        wires = self.map_wires(wires)
        return qml.math.reduce_dm(state, indices=wires, c_dtype=self.C_DTYPE)

    def purity(self, mp, **kwargs):
        """Returns the purity of the final state"""
        state = getattr(self, 'state', None)
        wires = self.map_wires(mp.wires)
        return qml.math.purity(state, indices=wires, c_dtype=self.C_DTYPE)

    def reset(self):
        """Resets the device"""
        super().reset()
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None
        rho = qnp.reshape(self._state, (2 ** self.num_wires, 2 ** self.num_wires))
        probs = self.marginal_prob(qnp.diagonal(rho), wires)
        probs = qnp.real(probs)
        return qnp.where(probs < 0, -probs, probs)

    def _get_kraus(self, operation):
        """Return the Kraus operators representing the operation.

        Args:
            operation (.Operation): a PennyLane operation

        Returns:
            list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
            the operation is unitary, returns a single Kraus operator. In the case of a diagonal
            unitary, returns a 1D array representing the matrix diagonal.
        """
        if operation in diagonal_in_z_basis:
            return operation.eigvals()
        if isinstance(operation, Channel):
            return operation.kraus_matrices()
        return [operation.matrix()]

    def _apply_channel(self, kraus, wires):
        """Apply a quantum channel specified by a list of Kraus operators to subsystems of the
        quantum state. For a unitary gate, there is a single Kraus operator.

        Args:
            kraus (list[array]): Kraus operators
            wires (Wires): target wires
        """
        channel_wires = self.map_wires(wires)
        rho_dim = 2 * self.num_wires
        num_ch_wires = len(channel_wires)
        kraus_dagger = [qnp.conj(qnp.transpose(k)) for k in kraus]
        kraus = qnp.stack(kraus)
        kraus_dagger = qnp.stack(kraus_dagger)
        kraus_shape = [len(kraus)] + [2] * num_ch_wires * 2
        kraus = qnp.cast(qnp.reshape(kraus, kraus_shape), dtype=self.C_DTYPE)
        kraus_dagger = qnp.cast(qnp.reshape(kraus_dagger, kraus_shape), dtype=self.C_DTYPE)
        state_indices = ABC[:rho_dim]
        row_wires_list = channel_wires.tolist()
        row_indices = ''.join(ABC_ARRAY[row_wires_list].tolist())
        col_wires_list = [w + self.num_wires for w in row_wires_list]
        col_indices = ''.join(ABC_ARRAY[col_wires_list].tolist())
        new_row_indices = ABC[rho_dim:rho_dim + num_ch_wires]
        new_col_indices = ABC[rho_dim + num_ch_wires:rho_dim + 2 * num_ch_wires]
        kraus_index = ABC[rho_dim + 2 * num_ch_wires:rho_dim + 2 * num_ch_wires + 1]
        new_state_indices = functools.reduce(lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]), zip(col_indices + row_indices, new_col_indices + new_row_indices), state_indices)
        einsum_indices = f'{kraus_index}{new_row_indices}{row_indices}, {state_indices},{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}'
        self._state = qnp.einsum(einsum_indices, kraus, self._state, kraus_dagger)

    def _apply_channel_tensordot(self, kraus, wires):
        """Apply a quantum channel specified by a list of Kraus operators to subsystems of the
        quantum state. For a unitary gate, there is a single Kraus operator.

        Args:
            kraus (list[array]): Kraus operators
            wires (Wires): target wires
        """
        channel_wires = self.map_wires(wires)
        num_ch_wires = len(channel_wires)
        kraus_shape = [2] * (num_ch_wires * 2)
        kraus = [qnp.cast(qnp.reshape(k, kraus_shape), dtype=self.C_DTYPE) for k in kraus]
        row_wires_list = channel_wires.tolist()
        col_wires_list = [w + self.num_wires for w in row_wires_list]
        channel_col_ids = list(range(num_ch_wires, 2 * num_ch_wires))
        axes_left = [channel_col_ids, row_wires_list]
        axes_right = [col_wires_list, channel_col_ids]

        def _conjugate_state_with(k):
            """Perform the double tensor product k @ self._state @ k.conj().
            The `axes_left` and `axes_right` arguments are taken from the ambient variable space
            and `axes_right` is assumed to incorporate the tensor product and the transposition
            of k.conj() simultaneously."""
            return qnp.tensordot(qnp.tensordot(k, self._state, axes_left), qnp.conj(k), axes_right)
        if len(kraus) == 1:
            _state = _conjugate_state_with(kraus[0])
        else:
            _state = qnp.sum(qnp.stack([_conjugate_state_with(k) for k in kraus]), axis=0)
        source_left = list(range(num_ch_wires))
        dest_left = row_wires_list
        source_right = list(range(-num_ch_wires, 0))
        dest_right = col_wires_list
        self._state = qnp.moveaxis(_state, source_left + source_right, dest_left + dest_right)

    def _apply_diagonal_unitary(self, eigvals, wires):
        """Apply a diagonal unitary gate specified by a list of eigenvalues. This method uses
        the fact that the unitary is diagonal for a more efficient implementation.

        Args:
            eigvals (array): eigenvalues (phases) of the diagonal unitary
            wires (Wires): target wires
        """
        channel_wires = self.map_wires(wires)
        eigvals = qnp.stack(eigvals)
        eigvals = qnp.cast(qnp.reshape(eigvals, [2] * len(channel_wires)), dtype=self.C_DTYPE)
        state_indices = ABC[:2 * self.num_wires]
        row_wires_list = channel_wires.tolist()
        row_indices = ''.join(ABC_ARRAY[row_wires_list].tolist())
        col_wires_list = [w + self.num_wires for w in row_wires_list]
        col_indices = ''.join(ABC_ARRAY[col_wires_list].tolist())
        einsum_indices = f'{row_indices},{state_indices},{col_indices}->{state_indices}'
        self._state = qnp.einsum(einsum_indices, eigvals, self._state, qnp.conj(eigvals))

    def _apply_basis_state(self, state, wires):
        """Initialize the device in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on
        """
        device_wires = self.map_wires(wires)
        n_basis_state = len(state)
        if not set(state).issubset({0, 1}):
            raise ValueError('BasisState parameter must consist of 0 or 1 integers.')
        if n_basis_state != len(device_wires):
            raise ValueError('BasisState parameter and wires must be of equal length.')
        basis_states = 2 ** (self.num_wires - 1 - device_wires.toarray())
        num = int(qnp.dot(state, basis_states))
        self._state = self._create_basis_state(num)

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state in a specified pure state.

        Args:
            state (array[complex]): normalized input state of length
                ``2**len(wires)``
            device_wires (Wires): wires that get initialized in the state
        """
        device_wires = self.map_wires(device_wires)
        state = qnp.asarray(state, dtype=self.C_DTYPE)
        n_state_vector = state.shape[0]
        if state.ndim != 1 or n_state_vector != 2 ** len(device_wires):
            raise ValueError('State vector must be of length 2**wires.')
        if not qnp.allclose(qnp.linalg.norm(state, ord=2), 1.0, atol=tolerance):
            raise ValueError('Sum of amplitudes-squared does not equal one.')
        if len(device_wires) == self.num_wires and sorted(device_wires.labels) == list(device_wires.labels):
            rho = qnp.outer(state, qnp.conj(state))
            self._state = qnp.reshape(rho, [2] * 2 * self.num_wires)
        else:
            basis_states = qnp.asarray(list(itertools.product([0, 1], repeat=len(device_wires))), dtype=int)
            unravelled_indices = qnp.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
            unravelled_indices[:, device_wires] = basis_states
            ravelled_indices = qnp.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
            state = qnp.scatter(ravelled_indices, state, [2 ** self.num_wires])
            rho = qnp.outer(state, qnp.conj(state))
            rho = qnp.reshape(rho, [2] * 2 * self.num_wires)
            self._state = qnp.asarray(rho, dtype=self.C_DTYPE)

    def _apply_density_matrix(self, state, device_wires):
        """Initialize the internal state in a specified mixed state.
        If not all the wires are specified in the full state :math:`\\rho`, remaining subsystem is filled by
        `\\mathrm{tr}_in(\\rho)`, which results in the full system state :math:`\\mathrm{tr}_{in}(\\rho) \\otimes \\rho_{in}`,
        where :math:`\\rho_{in}` is the argument `state` of this function and :math:`\\mathrm{tr}_{in}` is a partial
        trace over the subsystem to be replaced by this operation.

           Args:
               state (array[complex]): density matrix of length
                   ``(2**len(wires), 2**len(wires))``
               device_wires (Wires): wires that get initialized in the state
        """
        device_wires = self.map_wires(device_wires)
        state = qnp.asarray(state, dtype=self.C_DTYPE)
        state = qnp.reshape(state, (-1,))
        state_dim = 2 ** len(device_wires)
        dm_dim = state_dim ** 2
        if dm_dim != state.shape[0]:
            raise ValueError('Density matrix must be of length (2**wires, 2**wires)')
        if not qml.math.is_abstract(state) and (not qnp.allclose(qnp.trace(qnp.reshape(state, (state_dim, state_dim))), 1.0, atol=tolerance)):
            raise ValueError('Trace of density matrix is not equal one.')
        if len(device_wires) == self.num_wires and sorted(device_wires.labels) == list(device_wires.labels):
            self._state = qnp.reshape(state, [2] * 2 * self.num_wires)
            self._pre_rotated_state = self._state
        else:
            complement_wires = list(sorted(list(set(range(self.num_wires)) - set(device_wires))))
            sigma = self.density_matrix(Wires(complement_wires))
            rho = qnp.kron(sigma, state.reshape(state_dim, state_dim))
            rho = rho.reshape([2] * 2 * self.num_wires)
            left_axes = []
            right_axes = []
            complement_wires_count = len(complement_wires)
            for i in range(self.num_wires):
                if i in device_wires:
                    index = device_wires.index(i)
                    left_axes.append(complement_wires_count + index)
                    right_axes.append(complement_wires_count + index + self.num_wires)
                elif i in complement_wires:
                    index = complement_wires.index(i)
                    left_axes.append(index)
                    right_axes.append(index + self.num_wires)
            transpose_axes = left_axes + right_axes
            rho = qnp.transpose(rho, axes=transpose_axes)
            assert qml.math.is_abstract(rho) or qnp.allclose(qnp.trace(qnp.reshape(rho, (2 ** self.num_wires, 2 ** self.num_wires))), 1.0, atol=tolerance)
            self._state = qnp.asarray(rho, dtype=self.C_DTYPE)
            self._pre_rotated_state = self._state

    def _apply_operation(self, operation):
        """Applies operations to the internal device state.

        Args:
            operation (.Operation): operation to apply on the device
        """
        wires = operation.wires
        if operation.name == 'Identity':
            return
        if isinstance(operation, StatePrep):
            self._apply_state_vector(operation.parameters[0], wires)
            return
        if isinstance(operation, BasisState):
            self._apply_basis_state(operation.parameters[0], wires)
            return
        if isinstance(operation, QubitDensityMatrix):
            self._apply_density_matrix(operation.parameters[0], wires)
            return
        if isinstance(operation, Snapshot):
            if self._debugger and self._debugger.active:
                dim = 2 ** self.num_wires
                density_matrix = qnp.reshape(self._state, (dim, dim))
                if operation.tag:
                    self._debugger.snapshots[operation.tag] = density_matrix
                else:
                    self._debugger.snapshots[len(self._debugger.snapshots)] = density_matrix
            return
        matrices = self._get_kraus(operation)
        if operation in diagonal_in_z_basis:
            self._apply_diagonal_unitary(matrices, wires)
        else:
            num_op_wires = len(wires)
            interface = qml.math.get_interface(self._state, *matrices)
            if num_op_wires > 2 and interface in {'autograd', 'numpy'} or num_op_wires > 7:
                self._apply_channel_tensordot(matrices, wires)
            else:
                self._apply_channel(matrices, wires)

    def execute(self, circuit, **kwargs):
        """Execute a queue of quantum operations on the device and then
        measure the given observables.

        Applies a readout error to the measurement outcomes of any observable if
        readout_prob is non-zero. This is done by finding the list of measured wires on which
        BitFlip channels are applied in the :meth:`apply`.

        For plugin developers: instead of overwriting this, consider
        implementing a suitable subset of

        * :meth:`apply`

        * :meth:`~.generate_samples`

        * :meth:`~.probability`

        Additional keyword arguments may be passed to this method
        that can be utilised by :meth:`apply`. An example would be passing
        the ``QNode`` hash that can be used later for parametric compilation.

        Args:
            circuit (QuantumTape): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        if self.readout_err:
            wires_list = []
            for m in circuit.measurements:
                if isinstance(m, StateMP):
                    self.measured_wires = []
                    return super().execute(circuit, **kwargs)
                if isinstance(m, (SampleMP, CountsMP)) and m.wires in (qml.wires.Wires([]), self.wires):
                    self.measured_wires = self.wires
                    return super().execute(circuit, **kwargs)
                if isinstance(m, (VnEntropyMP, MutualInfoMP)):
                    continue
                wires_list.append(m.wires)
            self.measured_wires = qml.wires.Wires.all_wires(wires_list)
        return super().execute(circuit, **kwargs)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []
        for i, operation in enumerate(operations):
            if i > 0 and isinstance(operation, (StatePrep, BasisState)):
                raise DeviceError(f'Operation {operation.name} cannot be used after other Operations have already been applied on a {self.short_name} device.')
        for operation in operations:
            self._apply_operation(operation)
        self._pre_rotated_state = self._state
        for operation in rotations:
            self._apply_operation(operation)
        if self.readout_err:
            for k in self.measured_wires:
                bit_flip = qml.BitFlip(self.readout_err, wires=k)
                self._apply_operation(bit_flip)