from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
@simulator_tracking
@single_tape_support
class DefaultClifford(Device):
    """A PennyLane device for fast simulation of Clifford circuits using
    `stim <https://github.com/quantumlib/stim/>`_.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['aux_wire', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        check_clifford (bool): Check if all the gate operations in the circuits to be executed are Clifford. Default is ``True``.
        tableau (bool): Determines what should be returned when the device's state is computed with :func:`qml.state <pennylane.state>`.
            When ``True``, the device returns the final evolved Tableau. Alternatively, one may make it ``False`` to obtain
            the evolved state vector. Note that the latter might not be computationally feasible for larger qubit numbers.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from numpy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, try setting ``max_workers`` to ``None``.

    **Example:**

    .. code-block:: python

        dev = qml.device("default.clifford", tableau=True)

        @qml.qnode(dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.X(1)
            qml.ISWAP(wires=[0, 1])
            qml.Hadamard(wires=[0])
            return qml.state()

    >>> circuit()
    array([[0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1]])

    The devices execution pipeline can be investigated more closely with the following:

    .. code-block:: python

        num_qscripts = 5

        qscripts = [
            qml.tape.QuantumScript(
                [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
                [qml.expval(qml.Z(0))]
            )
        ] * num_qscripts

    >>> dev = DefaultClifford()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    (array(0), array(0), array(0), array(0), array(0))

    .. details::
        :title: Clifford Tableau
        :href: clifford-tableau-theory

        The device's internal state is represented by the following ``Tableau`` described in
        the `Sec. III, Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_:

        .. math::

            \\begin{bmatrix}
            x_{11} & \\cdots & x_{1n} &        & z_{11} & \\cdots & z_{1n} & &r_{1}\\\\
            \\vdots & \\ddots & \\vdots & & \\vdots & \\ddots & \\vdots & &\\vdots\\\\
            x_{n1} & \\cdots & x_{nn} &        & z_{n1} & \\cdots & z_{nn} & &r_{n}\\\\
            & & & & & & & & \\\\
            x_{\\left(  n+1\\right)  1} & \\cdots & x_{\\left(  n+1\\right)  n} & &
            z_{\\left(  n+1\\right)  1} & \\cdots & z_{\\left(  n+1\\right)  n} & & r_{n+1}\\\\
            \\vdots & \\ddots & \\vdots  & & \\vdots & \\ddots & \\vdots & & \\vdots\\\\
            x_{\\left(  2n\\right)  1}  & \\cdots & x_{\\left(  2n\\right)  n} & &
            z_{\\left(  2n\\right)  1}  & \\cdots & z_{\\left(  2n\\right)  n} & & r_{2n}
            \\end{bmatrix}

        The tableau's first `n` rows represent a destabilizer generator, while the
        remaining `n` rows represent the stabilizer generators. The Pauli representation
        for all of these generators are described using the :mod:`binary vector <pennylane.pauli.binary_to_pauli>`
        made from the binary variables :math:`x_{ij},\\ z_{ij}`,
        :math:`\\forall i\\in\\left\\{1,\\ldots,2n\\right\\}, j\\in\\left\\{1,\\ldots,n\\right\\}`
        and they together form the complete Pauli group.

        Finally, the last column of the tableau, with binary variables
        :math:`r_{i},\\ \\forall i\\in\\left\\{1,\\ldots,2n\\right\\}`,
        denotes whether the phase is negative (:math:`r_i = 1`) or not, for each generator.
        Maintaining and working with this tableau representation instead of the complete state vector
        makes the calculations of increasingly large Clifford circuits more efficient on this device.

    .. details::
        :title: Probabilities for Basis States
        :href: clifford-probabilities

        As the ``default.clifford`` device supports executing quantum circuits with a large number of qubits,
        the ability to compute the ``analytical`` probabilities for ``all`` computational basis states at
        once becomes computationally expensive and challenging as the system size increases. While we don't
        manually restrict users from doing so for any circuit, one can expect the underlying computation
        to reach its limit with ``20-24`` qubits on a typical consumer grade machine.

        As long as number of qubits are below this limit, one can simply use the :func:`qml.probs <pennylane.probs>`
        function with its usual arguments to compute probabilities for the complete computational basis states.
        We test this for a circuit that prepares the ``n``-qubit Greenberger-Horne-Zeilinger (GHZ) state.
        This means that the probabilities for the basis states :math:`|0\\rangle^{\\otimes n}` and
        :math:`|1\\rangle^{\\otimes n}` should be :math:`0.5`, and :math:`0.0` for the rest.

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            dev = qml.device("default.clifford")

            num_wires = 3
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=[0])
                for idx in range(num_wires):
                    qml.CNOT(wires=[idx, idx+1])
                return qml.probs()

        >>> circuit()
        tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], requires_grad=True)

        Once above the limit (or even otherwise), one can obtain the probability
        of a single target basis state by computing the expectation value of the
        corresponding projector using :mod:`qml.expval <pennylane.expval>` and
        :mod:`qml.Projector <pennylane.Projector>`.

        .. code-block:: python

            num_wires = 4
            @qml.qnode(dev)
            def circuit(state):
                qml.Hadamard(wires=[0])
                for idx in range(num_wires):
                    qml.CNOT(wires=[idx, idx+1])
                return qml.expval(qml.Projector(state, wires=range(num_wires)))

        >>> basis_states = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        >>> circuit(basis_states[0])
        tensor(0.5, requires_grad=True)
        >>> circuit(basis_states[1])
        tensor(0.0, requires_grad=True)
        >>> circuit(basis_states[2])
        tensor(0.0, requires_grad=True)

    .. details::
        :title: Error Channels
        :href: clifford-errors

        This device supports the finite-shot execution of quantum circuits with
        the following error channels that add Pauli noise, allowing for one to perform
        any sampling-based measurements.

        - *Multi-qubit Pauli errors:* :mod:`qml.PauliError <pennylane.PauliError>`
        - *Single-qubit depolarization errors:* :mod:`qml.DepolarizingChannel <pennylane.DepolarizingChannel>`
        - *Single-qubit flip errors:* :mod:`qml.BitFlip <pennylane.BitFlip>` and :mod:`qml.PhaseFlip <pennylane.PhaseFlip>`

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            dev = qml.device("default.clifford", shots=1024, seed=42)

            num_wires = 3
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=[0])
                for idx in range(num_wires):
                    qml.CNOT(wires=[idx, idx+1])
                qml.BitFlip(0.2, wires=[1])
                return qml.counts()

        >>> circuit()
        {'0000': 417, '0100': 95, '1011': 104, '1111': 408}

    .. details::
        :title: Tracking
        :href: clifford-tracking

        ``DefaultClifford`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions,
          such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`.

    .. details::
        :title: Accelerate calculations with multiprocessing
        :href: clifford-multiprocessing

        See the details in :class:`~pennylane.devices.DefaultQubit`'s "Accelerate calculations with multiprocessing"
        section. Additional information regarding multiprocessing can be found in the
        `multiprocessing docs page <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.
    """

    @property
    def name(self):
        """The name of the device."""
        return 'default.clifford'

    def __init__(self, wires=None, shots=None, check_clifford=True, tableau=True, seed='global', max_workers=None) -> None:
        if not has_stim:
            raise ImportError('This feature requires stim, a fast stabilizer circuit simulator. It can be installed with:\n\npip install stim')
        super().__init__(wires=wires, shots=shots)
        self._max_workers = max_workers
        self._check_clifford = check_clifford
        self._tableau = tableau
        seed = np.random.randint(0, high=10000000) if seed == 'global' else seed
        self._rng = np.random.default_rng(seed)
        self._debugger = None

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig)

        Returns:
            ExecutionConfig: a preprocessed execution config

        """
        updated_values = {}
        if execution_config.gradient_method == 'best':
            updated_values['gradient_method'] = None
        updated_values['use_device_jacobian_product'] = False
        if execution_config.grad_on_execution is None:
            updated_values['grad_on_execution'] = False
        updated_values['device_options'] = dict(execution_config.device_options)
        if 'max_workers' not in updated_values['device_options']:
            updated_values['device_options']['max_workers'] = self._max_workers
        if 'rng' not in updated_values['device_options']:
            updated_values['device_options']['rng'] = self._rng
        if 'tableau' not in updated_values['device_options']:
            updated_values['device_options']['tableau'] = self._tableau
        return replace(execution_config, **updated_values)

    def preprocess(self, execution_config: ExecutionConfig=DefaultExecutionConfig) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device currently does not intrinsically support parameter broadcasting.

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()
        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(qml.defer_measurements, device=self)
        if self._check_clifford:
            transform_program.add_transform(decompose, stopping_condition=operation_stopping_condition, name=self.name)
            transform_program.add_transform(_validate_channels, name=self.name)
        transform_program.add_transform(validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name)
        transform_program.add_transform(validate_observables, stopping_condition=observable_stopping_condition, name=self.name)
        max_workers = config.device_options.get('max_workers', self._max_workers)
        if max_workers:
            transform_program.add_transform(validate_multiprocessing_workers, max_workers, self)
        transform_program.add_transform(validate_adjoint_trainable_params)
        if config.gradient_method is not None:
            config.gradient_method = None
        return (transform_program, config)

    def execute(self, circuits: QuantumTape_or_Batch, execution_config: ExecutionConfig=DefaultExecutionConfig) -> Result_or_ResultBatch:
        max_workers = execution_config.device_options.get('max_workers', self._max_workers)
        if max_workers is None:
            seeds = self._rng.integers(2 ** 31 - 1, size=len(circuits))
            return tuple((self.simulate(c, seed=s, debugger=self._debugger) for c, s in zip(circuits, seeds)))
        vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
        seeds = self._rng.integers(2 ** 31 - 1, size=len(vanilla_circuits))
        _wrap_simulate = partial(self.simulate, debugger=None)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            exec_map = executor.map(_wrap_simulate, vanilla_circuits, seeds)
            results = tuple(exec_map)
        self._rng = np.random.default_rng(self._rng.integers(2 ** 31 - 1))
        return results

    def simulate(self, circuit: qml.tape.QuantumScript, seed=None, debugger=None) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            debugger (_Debugger): The debugger to use

        Returns:
            tuple(TensorLike): The results of the simulation

        This function assumes that all operations are Clifford.

        >>> qs = qml.tape.QuantumScript([qml.Hadamard(wires=0)], [qml.expval(qml.Z(0)), qml.state()])
        >>> qml.devices.DefaultClifford().simulate(qs)
        (array(0),
         array([[0, 1, 0],
                [1, 0, 0]]))

        """
        circuit = circuit.map_to_standard_wires()
        stim_circuit = stim.Circuit()
        tableau_simulator = stim.TableauSimulator()
        if self.wires is not None:
            tableau_simulator.set_num_qubits(len(self.wires))
        prep = None
        if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
            prep = circuit[0]
        use_prep_ops = bool(prep)
        if use_prep_ops:
            stim_tableau = stim.Tableau.from_state_vector(qml.math.reshape(prep.state_vector(wire_order=list(circuit.op_wires)), (1, -1))[0], endian='big')
            stim_circuit += stim_tableau.to_circuit()
        global_phase_ops = []
        for op in circuit.operations[use_prep_ops:]:
            gate, wires = _pl_op_to_stim(op)
            if gate is not None:
                stim_circuit.append_from_stim_program_text(f'{gate} {wires}')
            else:
                if op.name == 'GlobalPhase':
                    global_phase_ops.append(op)
                if op.name == 'Snapshot':
                    if debugger is not None and debugger.active:
                        meas = op.hyperparameters['measurement']
                        if meas is not None and (not isinstance(meas, qml.measurements.StateMP)):
                            raise ValueError(f'{self.name} does not support arbitrary measurements of a state with snapshots.')
                        snap_sim = stim.TableauSimulator()
                        if self.wires is not None:
                            snap_sim.set_num_qubits(len(self.wires))
                        snap_sim.do_circuit(stim_circuit)
                        state = self._measure_state(meas, snap_sim, circuit=circuit)
                        debugger.snapshots[op.tag or len(debugger.snapshots)] = state
        tableau_simulator.do_circuit(stim_circuit)
        global_phase = qml.GlobalPhase(qml.math.sum((op.data[0] for op in global_phase_ops)))
        if circuit.shots:
            meas_results = self.measure_statistical(circuit, stim_circuit, seed=seed)
        else:
            meas_results = self.measure_analytical(circuit, stim_circuit, tableau_simulator, global_phase)
        return meas_results[0] if len(meas_results) == 1 else tuple(meas_results)

    @staticmethod
    def _measure_observable_sample(meas_obs, stim_circuit, shots, sample_seed):
        """Compute sample output from a stim circuit for a given Pauli observable"""
        meas_dict = {'X': 'MX', 'Y': 'MY', 'Z': 'MZ', '_': 'M'}
        if isinstance(meas_obs, BasisStateProjector):
            stim_circ = stim_circuit.copy()
            stim_circ.append_from_stim_program_text('M ' + ' '.join(map(str, meas_obs.wires)))
            sampler = stim_circ.compile_sampler(seed=sample_seed)
            return ([qml.math.array(sampler.sample(shots=shots), dtype=int)], qml.math.array([1.0]))
        coeffs, paulis = _pl_obs_to_linear_comb(meas_obs)
        samples = []
        for pauli, wire in paulis:
            stim_circ = stim_circuit.copy()
            for op, wr in zip(pauli, wire):
                if op != 'I':
                    stim_circ.append(meas_dict[op], wr)
            sampler = stim_circ.compile_sampler(seed=sample_seed)
            samples.append(qml.math.array(sampler.sample(shots=shots), dtype=int))
        return (samples, qml.math.array(coeffs))

    def measure_statistical(self, circuit, stim_circuit, seed=None):
        """Given a circuit, compute samples and return the statistical measurement results."""
        num_shots = circuit.shots.total_shots
        sample_seed = seed if isinstance(seed, int) else self._rng.integers(2 ** 31 - 1, size=1)[0]
        measurement_map = {ExpectationMP: self._sample_expectation, VarianceMP: self._sample_variance, ClassicalShadowMP: self._sample_classical_shadow, ShadowExpvalMP: self._sample_expval_shadow}
        results = []
        for meas in circuit.measurements:
            measurement_func = measurement_map.get(type(meas), None)
            if measurement_func is not None:
                res = measurement_func(meas, stim_circuit, shots=num_shots, seed=sample_seed)
            else:
                meas_wires = meas.wires if meas.wires else range(stim_circuit.num_qubits)
                wire_order = {wire: idx for idx, wire in enumerate(meas.wires)}
                meas_op = meas.obs or qml.prod(*[qml.Z(idx) for idx in meas_wires])
                samples = self._measure_observable_sample(meas_op, stim_circuit, num_shots, sample_seed)[0]
                if len(samples) > 1:
                    raise qml.QuantumFunctionError(f'Observable {meas_op.name} is not supported for rotating probabilities on {self.name}.')
                res = meas.process_samples(samples=np.array(samples), wire_order=wire_order)
                if isinstance(meas, CountsMP):
                    res = res[0]
                elif isinstance(meas, SampleMP):
                    res = np.squeeze(res)
            results.append(res)
        return results

    def measure_analytical(self, circuit, stim_circuit, tableau_simulator, global_phase):
        """Given a circuit, compute tableau and return the analytical measurement results."""
        measurement_map = {DensityMatrixMP: self._measure_density_matrix, StateMP: self._measure_state, ExpectationMP: self._measure_expectation, VarianceMP: self._measure_variance, VnEntropyMP: self._measure_vn_entropy, MutualInfoMP: self._measure_mutual_info, PurityMP: self._measure_purity, ProbabilityMP: self._measure_probability}
        results = []
        for meas in circuit.measurements:
            measurement_func = measurement_map.get(type(meas), None)
            if measurement_func is None:
                raise NotImplementedError(f"default.clifford doesn't support the {type(meas)} measurement analytically at the moment.")
            results.append(measurement_func(meas, tableau_simulator, circuit=circuit, stim_circuit=stim_circuit, global_phase=global_phase))
        return results

    @staticmethod
    def _measure_density_matrix(meas, tableau_simulator, **_):
        """Measure the density matrix from the state of simulator device."""
        wires = list(meas.wires)
        state_vector = qml.math.array(tableau_simulator.state_vector(endian='big'))
        return qml.math.reduce_dm(qml.math.einsum('i, j->ij', state_vector, state_vector), wires)

    def _measure_state(self, _, tableau_simulator, **kwargs):
        """Measure the state of the simualtor device."""
        wires = kwargs.get('circuit').wires
        global_phase = kwargs.get('global_phase', qml.GlobalPhase(0.0))
        if self._tableau:
            tableau = tableau_simulator.current_inverse_tableau().inverse()
            x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
            pl_tableau = np.vstack((np.hstack((x2x, x2z, x_signs.reshape(-1, 1))), np.hstack((z2x, z2z, z_signs.reshape(-1, 1))))).astype(int)
            if pl_tableau.shape == (0, 1) and len(wires):
                return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            return pl_tableau
        state = qml.math.array(tableau_simulator.state_vector(endian='big'))
        if state.shape == (1,) and len(wires):
            state = qml.math.zeros(2 ** len(wires), dtype=complex)
            state[0] = 1.0 + 0j
        return state * qml.matrix(global_phase)[0][0]

    def _measure_expectation(self, meas, tableau_simulator, **kwargs):
        """Measure the expectation value with respect to the state of simulator device."""
        meas_obs = meas.obs
        if isinstance(meas_obs, BasisStateProjector):
            kwargs['prob_states'] = qml.math.array([meas_obs.data[0]])
            return self._measure_probability(qml.probs(wires=meas_obs.wires), tableau_simulator, **kwargs).squeeze()
        coeffs, paulis = _pl_obs_to_linear_comb(meas_obs)
        expecs = qml.math.zeros_like(coeffs)
        for idx, (pauli, wire) in enumerate(paulis):
            pauli_term = ['I'] * max(np.max(list(wire)) + 1, tableau_simulator.num_qubits)
            for op, wr in zip(pauli, wire):
                pauli_term[wr] = op
            stim_pauli = stim.PauliString(''.join(pauli_term))
            expecs[idx] = tableau_simulator.peek_observable_expectation(stim_pauli)
        return qml.math.dot(coeffs, expecs)

    def _measure_variance(self, meas, tableau_simulator, **_):
        """Measure the variance with respect to the state of simulator device."""
        meas_obs = qml.operation.convert_to_opmath(meas.obs)
        meas_obs1 = meas_obs.simplify()
        meas_obs2 = (meas_obs1 ** 2).simplify()
        return self._measure_expectation(ExpectationMP(meas_obs2), tableau_simulator) - self._measure_expectation(ExpectationMP(meas_obs1), tableau_simulator) ** 2

    def _measure_vn_entropy(self, meas, tableau_simulator, **kwargs):
        """Measure the Von Neumann entropy with respect to the state of simulator device."""
        wires = kwargs.get('circuit').wires
        tableau = tableau_simulator.current_inverse_tableau().inverse()
        z_stabs = qml.math.array([tableau.z_output(wire) for wire in range(len(wires))])
        return self._measure_stabilizer_entropy(z_stabs, list(meas.wires), meas.log_base)

    def _measure_mutual_info(self, meas, tableau_simulator, **kwargs):
        """Measure the mutual information between the subsystems of simulator device."""
        wires = kwargs.get('circuit').wires
        tableau = tableau_simulator.current_inverse_tableau().inverse()
        z_stabs = qml.math.array([tableau.z_output(wire) for wire in range(len(wires))])
        indices0, indices1 = getattr(meas, '_wires')
        return self._measure_stabilizer_entropy(z_stabs, list(indices0), meas.log_base) + self._measure_stabilizer_entropy(z_stabs, list(indices1), meas.log_base)

    def _measure_purity(self, meas, tableau_simulator, **kwargs):
        """Measure the purity of the state of simulator device.

        Computes the state purity using the monotonically decreasing second-order Rényi entropy
        form given in `Sci Rep 13, 4601 (2023) <https://www.nature.com/articles/s41598-023-31273-9>`_.
        We utilize the fact that Rényi entropies are equal for all Rényi indices ``n`` for the
        stabilizer states.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
        wires = kwargs.get('circuit').wires
        if wires == meas.wires:
            return qml.math.array(1.0)
        tableau = tableau_simulator.current_inverse_tableau().inverse()
        z_stabs = qml.math.array([tableau.z_output(wire) for wire in range(len(wires))])
        return 2 ** (-self._measure_stabilizer_entropy(z_stabs, list(meas.wires), log_base=2))

    @staticmethod
    def _measure_stabilizer_entropy(stabilizer, wires, log_base=None):
        """Computes the Rényi entanglement entropy using stabilizer information.

        Computes the Rényi entanglement entropy :math:`S_A` for a subsytem described
        by :math:`A`, :math:`S_A = \\text{rank}(\\text{proj}_A {\\mathcal{S}}) - |A|`,
        where :math:`\\mathcal{S}` is the stabilizer group for the system using the theory
        described in Appendix A.1.d of `arXiv:1901.08092 <https://arxiv.org/abs/1901.08092>`_.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
        num_qubits = qml.math.shape(stabilizer)[0]
        if len(wires) == num_qubits:
            return 0.0
        pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
        terms = [qml.pauli.PauliWord({idx: pauli_dict[ele] for idx, ele in enumerate(row)}) for row in stabilizer]
        binary_mat = qml.pauli.utils._binary_matrix_from_pws(terms, num_qubits)
        partition_mat = qml.math.hstack((binary_mat[:, num_qubits:][:, wires], binary_mat[:, :num_qubits][:, wires]))
        rank = qml.math.sum(qml.math.any(qml.qchem.tapering._reduced_row_echelon(partition_mat), axis=1))
        entropy = qml.math.log(2) * (rank - len(wires))
        if log_base is None:
            return entropy
        return entropy / qml.math.log(log_base)

    def _measure_probability(self, meas, _, **kwargs):
        """Measure the probability of each computational basis state.

        Computes the probability for each of the computational basis state vector iteratively
        according to the follow pseudocode.

        1. First, a complete basis set is built based on measured wires' length `l` by transforming
           integers :math:`[0, 2^l)` to their corresponding binary vector form, if the selective
           target states for computing probabilities have not been specified in the ``kwargs``.
        2. Second, We then build a `stim.TableauSimulator` based on the input circuit. If an observable
           `obs` is given, an additional diagonalizing circuit is appended to the input circuit for
           rotating the computational basis based on the `diagonalizing_gates` method of the observable.
        3. Finally, for every basis state, we iterate over each measured qubit `q_i` and peek if it can
           be collapsed to the state :math:`|0\\rangle` / :math:`\\rangle`1` corresponding to the bit
           `0` / `1` in the basis state vector at `i`th index.
        4. If the qubit can be collapsed to the correct state, we do the post-selection and continue. If not,
           we identify it as an `unattainable` state and assign them with a zero probability. We do so for all
           the other basis states with the same `i`th index and keep this information stored in a visit-array.
        5. Alternatively, if the qubit is in a superposition state, then it can collapse to either of the states
           :math:`|0\\rangle` / :math:`\\rangle`1`. We identify this as a `branching` scenario. We half the
           current probability and post-select based on the `i`th index of the basis state we are iterating.
        """
        circuit = kwargs.get('circuit')
        tgt_states = kwargs.get('prob_states', None)
        mobs_wires = meas.obs.wires if meas.obs else meas.wires
        meas_wires = mobs_wires if mobs_wires else circuit.wires
        if tgt_states is None:
            num_wires = len(meas_wires)
            basis_vec = np.arange(2 ** num_wires)[:, np.newaxis]
            tgt_states = (basis_vec & 1 << np.arange(num_wires)[::-1] > 0).astype(int)
        diagonalizing_cit = kwargs.get('stim_circuit').copy()
        diagonalizing_ops = [] if not meas.obs else meas.obs.diagonalizing_gates()
        for diag_op in diagonalizing_ops:
            if diag_op.name not in _OPERATIONS_MAP:
                raise ValueError(f'Currently, we only support observables whose diagonalizing gates are Clifford, got {diag_op}')
            stim_op = _pl_op_to_stim(diag_op)
            if stim_op[0] is not None:
                diagonalizing_cit.append_from_stim_program_text(f'{stim_op[0]} {stim_op[1]}')
        circuit_simulator = stim.TableauSimulator()
        circuit_simulator.do_circuit(diagonalizing_cit)
        if not self._tableau:
            state = self._measure_state(meas, circuit_simulator, circuit=circuit)
            return meas.process_state(state, wire_order=circuit.wires)
        if len(meas_wires) >= tgt_states.shape[1]:
            meas_wires = meas_wires[:tgt_states.shape[1]]
        else:
            cgc_states = []
            for state in tgt_states:
                if list(state[meas_wires]) not in cgc_states:
                    cgc_states.append(list(state[meas_wires]))
            tgt_states = np.array(cgc_states)
        tgt_integs = np.array([int(''.join(map(str, tgt_state)), 2) for tgt_state in tgt_states])
        visited_probs = []
        prob_res = np.ones(tgt_states.shape[0])
        for tgt_index, (tgt_integ, tgt_state) in enumerate(zip(tgt_integs, tgt_states)):
            if tgt_integ in visited_probs:
                continue
            prob_sim = circuit_simulator.copy()
            for idx, wire in enumerate(meas_wires):
                expectation = prob_sim.peek_z(wire)
                outcome = int(0.5 * (1 - expectation))
                if not expectation:
                    prob_res[tgt_index] /= 2.0
                else:
                    nope_idx = np.where(np.squeeze(np.all(tgt_states[:, :idx] == tgt_state[:idx], axis=-1)) & tgt_states[:, idx] != outcome)[0] if idx else np.where(tgt_states[:, idx] != outcome)[0]
                    nope_idx = np.setdiff1d(nope_idx, visited_probs)
                    prob_res[nope_idx] = 0.0
                    visited_probs.extend(tgt_integs[nope_idx])
                    if tgt_state[idx] != outcome:
                        prob_res[tgt_index] = 0.0
                        break
                prob_sim.postselect_z(wire, desired_value=tgt_state[idx])
            visited_probs.append(tgt_integ)
        return prob_res

    def _sample_expectation(self, meas, stim_circuit, shots, seed):
        """Measure the expectation value with respect to samples from simulator device."""
        meas_op = meas.obs
        samples, coeffs = self._measure_observable_sample(meas_op, stim_circuit, shots, seed)
        if isinstance(meas_op, BasisStateProjector):
            matches = np.where((samples[0] == meas_op.data[0]).all(axis=1))[0]
            return len(matches) / shots
        expecs = [qml.math.mean(qml.math.power([-1] * shots, qml.math.sum(sample, axis=1))) for sample in samples]
        return qml.math.dot(coeffs, expecs)

    def _sample_variance(self, meas, stim_circuit, shots, seed):
        """Measure the variance with respect to samples from simulator device."""
        meas_op = meas.obs
        meas_obs = qml.operation.convert_to_opmath(meas_op)
        meas_obs1 = meas_obs.simplify()
        meas_obs2 = (meas_obs1 ** 2).simplify()
        return self._sample_expectation(qml.expval(meas_obs2), stim_circuit, shots, seed) - self._sample_expectation(qml.expval(meas_obs1), stim_circuit, shots, seed) ** 2

    @staticmethod
    def _measure_single_sample(stim_ct, meas_ops, meas_idx, meas_wire):
        """Sample a single qubit Pauli measurement from a stim circuit"""
        stim_sm = stim.TableauSimulator()
        stim_sm.do_circuit(stim_ct)
        return stim_sm.measure_observable(stim.PauliString([0] * meas_idx + meas_ops + [0] * (meas_wire - meas_idx - 1)))

    def _sample_classical_shadow(self, meas, stim_circuit, shots, seed):
        """Measures classical shadows from the state of simulator device"""
        meas_seed = meas.seed or seed
        meas_wire = stim_circuit.num_qubits
        bits = []
        recipes = np.random.RandomState(meas_seed).randint(3, size=(shots, meas_wire))
        for recipe in recipes:
            bits.append([self._measure_single_sample(stim_circuit, [rec + 1], idx, meas_wire) for idx, rec in enumerate(recipe)])
        return (np.asarray(bits, dtype=int), np.asarray(recipes, dtype=int))

    def _sample_expval_shadow(self, meas, stim_circuit, shots, seed):
        """Measures expectation value of a Pauli observable using
        classical shadows from the state of simulator device."""
        bits, recipes = self._sample_classical_shadow(meas, stim_circuit, shots, seed)
        wires_map = list(range(stim_circuit.num_qubits))
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=wires_map)
        return shadow.expval(meas.H, meas.k)