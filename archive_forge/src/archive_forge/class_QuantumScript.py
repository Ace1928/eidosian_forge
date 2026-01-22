import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
class QuantumScript:
    """The operations and measurements that represent instructions for
    execution on a quantum device.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed

    Keyword Args:
        shots (None, int, Sequence[int], ~.Shots): Number and/or batches of shots for execution.
            Note that this property is still experimental and under development.
        trainable_params (None, Sequence[int]): the indices for which parameters are trainable
        _update=True (bool): Whether or not to set various properties on initialization. Setting
            ``_update=False`` reduces computations if the script is only an intermediary step.

    .. seealso:: :class:`pennylane.tape.QuantumTape`

    **Example:**

    .. code-block:: python

        from pennylane.tape import QuantumScript

        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a")),
               qml.RX(0.432, 0),
               qml.RY(0.543, 0),
               qml.CNOT((0,"a")),
               qml.RX(0.133, "a")]

        qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

    >>> list(qscript)
    [BasisState(array([1, 1]), wires=[0, "a"]),
    RX(0.432, wires=[0]),
    RY(0.543, wires=[0]),
    CNOT(wires=[0, 'a']),
    RX(0.133, wires=['a']),
    expval(Z(0))]
    >>> qscript.operations
    [BasisState(array([1, 1]), wires=[0, "a"]),
    RX(0.432, wires=[0]),
    RY(0.543, wires=[0]),
    CNOT(wires=[0, 'a']),
    RX(0.133, wires=['a'])]
    >>> qscript.measurements
    [expval(Z(0))]

    Iterating over the quantum script can be done by:

    >>> for op in qscript:
    ...     print(op)
    BasisState(array([1, 1]), wires=[0, "a"])
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(Z(0))'

    Quantum scripts also support indexing and length determination:

    >>> qscript[0]
    BasisState(array([1, 1]), wires=[0, "a"])
    >>> len(qscript)
    6

    Once constructed, the script can be executed directly on a quantum device
    using the :func:`~.pennylane.execute` function:

    >>> dev = qml.device('default.qubit', wires=(0,'a'))
    >>> qml.execute([qscript], dev, gradient_fn=None)
    [array([-0.77750694])]

    Quantum scripts can also store information about the number and batches of
    executions by setting the ``shots`` keyword argument. This information is internally
    stored in a :class:`pennylane.measurements.Shots` object:

    >>> s_vec = [1, 1, 2, 2, 2]
    >>> qscript = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.Z(0))], shots=s_vec)
    >>> qscript.shots.shot_vector
    (ShotCopies(1 shots x 2), ShotCopies(2 shots x 3))

    ``ops`` and ``measurements`` are converted to lists upon initialization,
    so those arguments accept any iterable object:

    >>> qscript = QuantumScript((qml.X(i) for i in range(3)))
    >>> qscript.circuit
    [X(0), X(1), X(2)]

    """

    def _flatten(self):
        return ((self._ops, self.measurements), (self.shots, tuple(self.trainable_params)))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, shots=metadata[0], trainable_params=metadata[1])

    def __init__(self, ops=None, measurements=None, shots: Optional[Union[int, Sequence, Shots]]=None, trainable_params: Optional[Sequence[int]]=None, _update=True):
        self._ops = [] if ops is None else list(ops)
        self._measurements = [] if measurements is None else list(measurements)
        self._shots = Shots(shots)
        self._par_info = []
        'list[dict[str, Operator or int]]: Parameter information.\n        Values are dictionaries containing the corresponding operation and operation parameter index.'
        self._trainable_params = trainable_params
        self._graph = None
        self._specs = None
        self._output_dim = None
        self._batch_size = _UNSET_BATCH_SIZE
        self.wires = _empty_wires
        self.num_wires = 0
        self._obs_sharing_wires = []
        'list[.Observable]: subset of the observables that share wires with another observable,\n        i.e., that do not have their own unique set of wires.'
        self._obs_sharing_wires_id = []
        if _update:
            self._update()

    def __repr__(self):
        return f'<{self.__class__.__name__}: wires={self.wires.tolist()}, params={self.num_params}>'

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the quantum script"""
        fingerprint = []
        fingerprint.extend((op.hash for op in self.operations))
        fingerprint.extend((m.hash for m in self.measurements))
        fingerprint.extend(self.trainable_params)
        fingerprint.extend(self.shots)
        return hash(tuple(fingerprint))

    def __iter__(self):
        """list[.Operator, .MeasurementProcess]: Return an iterator to the
        underlying quantum circuit object."""
        return iter(self.circuit)

    def __getitem__(self, idx):
        """list[.Operator]: Return the indexed operator from underlying quantum
        circuit object."""
        return self.circuit[idx]

    def __len__(self):
        """int: Return the number of operations and measurements in the
        underlying quantum circuit object."""
        return len(self.circuit)

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum script (if any)"""
        return None

    @property
    def circuit(self):
        """Returns the underlying quantum circuit as a list of operations and measurements.

        The circuit is created with the assumptions that:

        * The ``operations`` attribute contains quantum operations and
          mid-circuit measurements and
        * The ``measurements`` attribute contains terminal measurements.

        Note that the resulting list could contain MeasurementProcess objects
        that some devices may not support.

        Returns:

            list[.Operator, .MeasurementProcess]: the quantum circuit
            containing quantum operations and measurements
        """
        return self.operations + self.measurements

    @property
    def operations(self) -> List[Operator]:
        """Returns the state preparations and operations on the quantum script.

        Returns:
            list[.Operator]: quantum operations

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.operations
        [StatePrep([0, 1], wires=[0]), RX(0.432, wires=[0])]
        """
        return self._ops

    @property
    def observables(self) -> List[Union[MeasurementProcess, Observable]]:
        """Returns the observables on the quantum script.

        Returns:
            list[.MeasurementProcess, .Observable]]: list of observables

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.observables
        [expval(Z(0))]
        """
        obs = []
        for m in self.measurements:
            if m.obs is not None:
                obs.append(m.obs)
            else:
                obs.append(m)
        return obs

    @property
    def measurements(self) -> List[MeasurementProcess]:
        """Returns the measurements on the quantum script.

        Returns:
            list[.MeasurementProcess]: list of measurement processes

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.measurements
        [expval(Z(0))]
        """
        return self._measurements

    @property
    def samples_computational_basis(self):
        """Determines if any of the measurements are in the computational basis."""
        return any((o.samples_computational_basis for o in self.measurements))

    @property
    def num_params(self):
        """Returns the number of trainable parameters on the quantum script."""
        return len(self.trainable_params)

    @property
    def batch_size(self):
        """The batch size of the quantum script inferred from the batch sizes
        of the used operations for parameter broadcasting.

        .. seealso:: :attr:`~.Operator.batch_size` for details.

        Returns:
            int or None: The batch size of the quantum script if present, else ``None``.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._update_batch_size()
        return self._batch_size

    @property
    def output_dim(self):
        """The (inferred) output dimension of the quantum script."""
        if self._output_dim is None:
            self._update_output_dim()
        return self._output_dim

    @property
    def diagonalizing_gates(self) -> List[Operation]:
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[~.Operation]: the operations that diagonalize the observables
        """
        rotation_gates = []
        with qml.queuing.QueuingManager.stop_recording():
            for observable in self.observables:
                with contextlib.suppress(qml.operation.DiagGatesUndefinedError):
                    rotation_gates.extend(observable.diagonalizing_gates())
        return rotation_gates

    @property
    def shots(self) -> Shots:
        """Returns a ``Shots`` object containing information about the number
        and batches of shots

        Returns:
            ~.Shots: Object with shot information
        """
        return self._shots

    @property
    def num_preps(self) -> int:
        """Returns the index of the first operator that is not an StatePrepBase operator."""
        idx = 0
        num_ops = len(self.operations)
        while idx < num_ops and isinstance(self.operations[idx], qml.operation.StatePrepBase):
            idx += 1
        return idx

    @property
    def op_wires(self) -> Wires:
        """Returns the wires that the tape operations act on."""
        return Wires.all_wires((op.wires for op in self.operations))

    def _update(self):
        """Update all internal metadata regarding processed operations and observables"""
        self._graph = None
        self._specs = None
        self._update_circuit_info()
        self._update_par_info()
        self._update_observables()

    def _update_circuit_info(self):
        """Update circuit metadata

        Sets:
            wires (~.Wires): Wires
            num_wires (int): Number of wires
        """
        self.wires = Wires.all_wires(dict.fromkeys((op.wires for op in self)))
        self.num_wires = len(self.wires)

    def _update_par_info(self):
        """Update the parameter information list. Each entry in the list with an operation and an index
        into that operation's data.

        Sets:
            _par_info (list): Parameter information
        """
        self._par_info = []
        for idx, op in enumerate(self.operations):
            self._par_info.extend(({'op': op, 'op_idx': idx, 'p_idx': i} for i, d in enumerate(op.data)))
        n_ops = len(self.operations)
        for idx, m in enumerate(self.measurements):
            if m.obs is not None:
                self._par_info.extend(({'op': m.obs, 'op_idx': idx + n_ops, 'p_idx': i} for i, d in enumerate(m.obs.data)))

    def _update_observables(self):
        """Update information about observables, including the wires that are acted upon and
        identifying any observables that share wires.

        Sets:
            _obs_sharing_wires (list[~.Observable]): Observables that share wires with
                any other observable
            _obs_sharing_wires_id (list[int]): Indices of the measurements that contain
                the observables in _obs_sharing_wires
        """
        obs_wires = [wire for m in self.measurements for wire in m.wires if m.obs is not None]
        self._obs_sharing_wires = []
        self._obs_sharing_wires_id = []
        if len(obs_wires) != len(set(obs_wires)):
            c = Counter(obs_wires)
            repeated_wires = {w for w in obs_wires if c[w] > 1}
            for i, m in enumerate(self.measurements):
                if m.obs is not None and len(set(m.wires) & repeated_wires) > 0:
                    self._obs_sharing_wires.append(m.obs)
                    self._obs_sharing_wires_id.append(i)

    def _update_batch_size(self):
        """Infer the batch_size of the quantum script from the batch sizes of its operations
        and check the latter for consistency.

        Sets:
            _batch_size (int): The common batch size of the quantum script operations, if any has one
        """
        candidate = None
        for op in self.operations:
            op_batch_size = getattr(op, 'batch_size', None)
            if op_batch_size is None:
                continue
            if candidate:
                if op_batch_size != candidate:
                    raise ValueError(f'The batch sizes of the quantum script operations do not match, they include {candidate} and {op_batch_size}.')
            else:
                candidate = op_batch_size
        self._batch_size = candidate

    def _update_output_dim(self):
        """Update the dimension of the output of the quantum script.

        Sets:
            self._output_dim (int): Size of the quantum script output (when flattened)

        This method makes use of `self.batch_size`, so that `self._batch_size`
        needs to be up to date when calling it.
        Call `_update_batch_size` before `_update_output_dim`
        """
        self._output_dim = 0
        for m in self.measurements:
            if isinstance(m, ProbabilityMP):
                self._output_dim += 2 ** len(m.wires)
            elif not isinstance(m, StateMP):
                self._output_dim += 1
        if self.batch_size:
            self._output_dim *= self.batch_size

    @property
    def data(self):
        """Alias to :meth:`~.get_parameters` and :meth:`~.set_parameters`
        for backwards compatibilities with operations."""
        return self.get_parameters(trainable_only=False)

    @property
    def trainable_params(self):
        """Store or return a list containing the indices of parameters that support
        differentiability. The indices provided match the order of appearence in the
        quantum circuit.

        Setting this property can help reduce the number of quantum evaluations needed
        to compute the Jacobian; parameters not marked as trainable will be
        automatically excluded from the Jacobian computation.

        The number of trainable parameters determines the number of parameters passed to
        :meth:`~.set_parameters`, and changes the default output size of method :meth:`~.get_parameters()`.

        .. note::

            For devices that support native backpropagation (such as
            ``default.qubit.tf`` and ``default.qubit.autograd``), this
            property contains no relevant information when using
            backpropagation to compute gradients.

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.trainable_params
        [0, 1, 2]
        >>> qscript.trainable_params = [0] # set only the first parameter as trainable
        >>> qscript.get_parameters()
        [0.432]
        """
        if self._trainable_params is None:
            self._trainable_params = list(range(len(self._par_info)))
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (list[int]): parameter indices
        """
        if any((not isinstance(i, int) or i < 0 for i in param_indices)):
            raise ValueError('Argument indices must be non-negative integers.')
        num_params = len(self._par_info)
        if any((i > num_params for i in param_indices)):
            raise ValueError(f'Quantum Script only has {num_params} parameters.')
        self._trainable_params = sorted(set(param_indices))

    def get_operation(self, idx):
        """Returns the trainable operation, the operation index and the corresponding operation argument
        index, for a specified trainable parameter index.

        Args:
            idx (int): the trainable parameter index

        Returns:
            tuple[.Operation, int, int]: tuple containing the corresponding
            operation, operation index and an integer representing the argument index,
            for the provided trainable parameter.
        """
        t_idx = self.trainable_params[idx]
        info = self._par_info[t_idx]
        return (info['op'], info['op_idx'], info['p_idx'])

    def get_parameters(self, trainable_only=True, operations_only=False, **kwargs):
        """Return the parameters incident on the quantum script operations.

        The returned parameters are provided in order of appearance
        on the quantum script.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            operations_only (bool): if True, returns only the parameters of the
                operations excluding parameters to observables of measurements

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

        By default, all parameters are trainable and will be returned:

        >>> qscript.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> qscript.trainable_params = [1] # set the second parameter as trainable
        >>> qscript.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> qscript.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        if trainable_only:
            params = []
            for p_idx in self.trainable_params:
                par_info = self._par_info[p_idx]
                if operations_only and isinstance(self[par_info['op_idx']], MeasurementProcess):
                    continue
                op = par_info['op']
                op_idx = par_info['p_idx']
                params.append(op.data[op_idx])
            return params
        params = [d for op in self.operations for d in op.data]
        if operations_only:
            return params
        for m in self.measurements:
            if m.obs is not None:
                params.extend(m.obs.data)
        return params

    def bind_new_parameters(self, params: Sequence[TensorLike], indices: Sequence[int]):
        """Create a new tape with updated parameters.

        This function takes a list of new parameters as input, and returns
        a new :class:`~.tape.QuantumScript` containing the new parameters at the provided indices,
        with the parameters at all other indices remaining the same.

        Args:
            params (Sequence[TensorLike]): New parameters to create the tape with. This
                must have the same length as ``indices``.
            indices (Sequence[int]): The parameter indices to update with the given parameters.
                The index of a parameter is defined as its index in ``tape.get_parameters()``.

        Returns:
            .tape.QuantumScript: New tape with updated parameters

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

        A new tape can be created by passing new parameters along with the indices
        to be updated. To modify all parameters in the above qscript:

        >>> new_qscript = qscript.bind_new_parameters([0.1, 0.2, 0.3], [0, 1, 2])
        >>> new_qscript.get_parameters()
        [0.1, 0.2, 0.3]

        The original ``qscript`` remains unchanged:

        >>> qscript.get_parameters()
        [0.432, 0.543, 0.133]

        A subset of parameters can be modified as well, defined by the parameter indices:

        >>> newer_qscript = new_qscript.bind_new_parameters([-0.1, 0.5], [0, 2])
        >>> newer_qscript.get_parameters()
        [-0.1, 0.2, 0.5]
        """
        if len(params) != len(indices):
            raise ValueError('Number of provided parameters does not match number of indices')
        op_indices = {}
        for param_idx, idx in enumerate(sorted(indices)):
            pinfo = self._par_info[idx]
            op_idx, p_idx = (pinfo['op_idx'], pinfo['p_idx'])
            if op_idx not in op_indices:
                op_indices[op_idx] = {}
            op_indices[op_idx][p_idx] = param_idx
        new_ops = self.circuit
        for op_idx, p_indices in op_indices.items():
            op = new_ops[op_idx]
            data = op.data if isinstance(op, Operator) else op.obs.data
            new_params = [params[p_indices[i]] if i in p_indices else d for i, d in enumerate(data)]
            if isinstance(op, Operator):
                new_op = qml.ops.functions.bind_new_parameters(op, new_params)
            else:
                new_obs = qml.ops.functions.bind_new_parameters(op.obs, new_params)
                new_op = op.__class__(obs=new_obs)
            new_ops[op_idx] = new_op
        new_operations = new_ops[:len(self.operations)]
        new_measurements = new_ops[len(self.operations):]
        return self.__class__(new_operations, new_measurements, shots=self.shots, trainable_params=self.trainable_params)

    def shape(self, device):
        """Produces the output shape of the quantum script by inspecting its measurements
        and the device used for execution.

        .. note::

            The computed shape is not stored because the output shape may be
            dependent on the device used for execution.

        Args:
            device (pennylane.Device): the device that will be used for the script execution

        Returns:
            Union[tuple[int], tuple[tuple[int]]]: the output shape(s) of the quantum script result

        **Examples**

        .. code-block:: pycon

            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.shape(dev)
            (4,)
            >>> m = [qml.state(), qml.expval(qml.Z(0)), qml.probs((0,1))]
            >>> qs = QuantumScript(measurements=m)
            >>> qs.shape(dev)
            ((4,), (), (4,))
        """
        if isinstance(device, qml.devices.Device):
            shots = self.shots
        else:
            shots = Shots(device._raw_shot_sequence) if device.shot_vector is not None else Shots(device.shots)
        if len(shots.shot_vector) > 1 and self.batch_size is not None:
            raise NotImplementedError('Parameter broadcasting when using a shot vector is not supported yet.')
        shapes = tuple((meas_process.shape(device, shots) for meas_process in self.measurements))
        if self.batch_size is not None:
            shapes = tuple(((self.batch_size,) + shape for shape in shapes))
        if len(shapes) == 1:
            return shapes[0]
        if shots.num_copies > 1:
            shapes = tuple(zip(*shapes))
        return shapes

    @property
    def numeric_type(self):
        """Returns the expected numeric type of the quantum script result by inspecting
        its measurements.

        Returns:
            Union[type, Tuple[type]]: The numeric type corresponding to the result type of the
            quantum script, or a tuple of such types if the script contains multiple measurements

        **Example:**

        .. code-block:: pycon

            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.numeric_type
            complex
        """
        types = tuple((observable.numeric_type for observable in self.measurements))
        return types[0] if len(types) == 1 else types

    def copy(self, copy_operations=False):
        """Returns a shallow copy of the quantum script.

        Args:
            copy_operations (bool): If True, the operations are also shallow copied.
                Otherwise, if False, the copied operations will simply be references
                to the original operations; changing the parameters of one script will likewise
                change the parameters of all copies.

        Returns:
            QuantumScript : a shallow copy of the quantum script
        """
        if copy_operations:
            _ops = [copy.copy(op) for op in self.operations]
            _measurements = [copy.copy(op) for op in self.measurements]
        else:
            _ops = self.operations.copy()
            _measurements = self.measurements.copy()
        new_qscript = self.__class__(ops=_ops, measurements=_measurements, shots=self.shots, trainable_params=list(self.trainable_params))
        new_qscript._graph = None if copy_operations else self._graph
        new_qscript._specs = None
        new_qscript.wires = copy.copy(self.wires)
        new_qscript.num_wires = self.num_wires
        new_qscript._update_par_info()
        new_qscript._obs_sharing_wires = self._obs_sharing_wires
        new_qscript._obs_sharing_wires_id = self._obs_sharing_wires_id
        new_qscript._batch_size = self._batch_size
        new_qscript._output_dim = self._output_dim
        return new_qscript

    def __copy__(self):
        return self.copy(copy_operations=True)

    def expand(self, depth=1, stop_at=None, expand_measurements=False):
        """Expand all operations to a specific depth.

        Args:
            depth (int): the depth the script should be expanded
            stop_at (Callable): A function which accepts a queue object,
                and returns ``True`` if this object should *not* be expanded.
                If not provided, all objects that support expansion will be expanded.
            expand_measurements (bool): If ``True``, measurements will be expanded
                to basis rotations and computational basis measurements.

        **Example**

        Consider the following nested quantum script:

        >>> nested_script = QuantumScript([qml.Rot(0.543, 0.1, 0.4, wires=0)])
        >>> ops = [
                qml.BasisState(np.array([1, 1]), wires=[0, 'a']),
                nested_script,
                qml.CNOT(wires=[0, 'a']), qml.RY(0.2, wires='a'),
            ]
        >>> measurements = [qml.probs(wires=0), qml.probs(wires='a')]
        >>> qscript = QuantumScript(ops, measurements)

        The nested structure is preserved:

        >>> qscript.operations
        [BasisState(tensor([1, 1], requires_grad=True), wires=[0, 'a']),
        <QuantumScript: wires=[0], params=3>,
        CNOT(wires=[0, 'a']),
        RY(0.2, wires=['a'])]

        Calling ``.expand`` will return a script with all nested scripts
        expanded, resulting in a single script of quantum operations:

        >>> new_qscript = qscript.expand(depth=2)
        >>> new_qscript.operations
        [X(0),
        X('a'),
        RZ(0.543, wires=[0]),
        RY(0.1, wires=[0]),
        RZ(0.4, wires=[0]),
        CNOT(wires=[0, 'a']),
        RY(0.2, wires=['a'])]
        """
        new_script = qml.tape.tape.expand_tape(self, depth=depth, stop_at=stop_at, expand_measurements=expand_measurements)
        new_script._update()
        return new_script

    def adjoint(self):
        """Create a quantum script that is the adjoint of this one.

        Adjointed quantum scripts are the conjugated and transposed version of the
        original script. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        Returns:
            ~.QuantumScript: the adjointed script
        """
        ops = self.operations[self.num_preps:]
        prep = self.operations[:self.num_preps]
        with qml.QueuingManager.stop_recording():
            ops_adj = [qml.adjoint(op, lazy=False) for op in reversed(ops)]
        return self.__class__(ops=prep + ops_adj, measurements=self.measurements, shots=self.shots)

    @property
    def graph(self):
        """Returns a directed acyclic graph representation of the recorded
        quantum circuit:

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.graph
        <pennylane.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .CircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = qml.CircuitGraph(self.operations, self.measurements, self.wires, self._par_info, self.trainable_params)
        return self._graph

    @property
    def specs(self):
        """Resource information about a quantum circuit.

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain quantum script specifications

        **Example**
         >>> ops = [qml.Hadamard(0), qml.RX(0.26, 1), qml.CNOT((1,0)),
         ...         qml.Rot(1.8, -2.7, 0.2, 0), qml.Hadamard(1), qml.CNOT((0, 1))]
         >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0) @ qml.Z(1))])

        Asking for the specs produces a dictionary of useful information about the circuit:

        >>> qscript.specs['num_observables']
        1
        >>> qscript.specs['gate_sizes']
        defaultdict(<class 'int'>, {1: 4, 2: 2})
        >>> print(qscript.specs['resources'])
        wires: 2
        gates: 6
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 2, 'RX': 1, 'CNOT': 2, 'Rot': 1}
        gate_sizes:
        {1: 4, 2: 2}
        """
        if self._specs is None:
            resources = qml.resource.resource._count_resources(self)
            self._specs = {'resources': resources, 'num_observables': len(self.observables), 'num_diagonalizing_gates': len(self.diagonalizing_gates), 'num_trainable_params': self.num_params}
        return self._specs

    def draw(self, wire_order=None, show_all_wires=False, decimals=None, max_length=100, show_matrices=True):
        """Draw the quantum script as a circuit diagram. See :func:`~.drawer.tape_text` for more information.

        Args:
            wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
            show_all_wires (bool): If True, all wires, including empty wires, are printed.
            decimals (int): How many decimal points to include when formatting operation parameters.
                Default ``None`` will omit parameters from operation labels.
            max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
                begin anew beneath the previous lines.
            show_matrices=True (bool): show matrix valued parameters below all circuit diagrams

        Returns:
            str: the circuit representation of the quantum script
        """
        return qml.drawer.tape_text(self, wire_order=wire_order, show_all_wires=show_all_wires, decimals=decimals, max_length=max_length, show_matrices=show_matrices)

    def to_openqasm(self, wires=None, rotations=True, measure_all=True, precision=None):
        """Serialize the circuit as an OpenQASM 2.0 program.

        Measurements are assumed to be performed on all qubits in the computational basis. An
        optional ``rotations`` argument can be provided so that output of the OpenQASM circuit is
        diagonal in the eigenbasis of the quantum script's observables. The measurement outputs can be
        restricted to only those specified in the script by setting ``measure_all=False``.

        .. note::

            The serialized OpenQASM program assumes that gate definitions
            in ``qelib1.inc`` are available.

        Args:
            wires (Wires or None): the wires to use when serializing the circuit
            rotations (bool): in addition to serializing user-specified
                operations, also include the gates that diagonalize the
                measured wires such that they are in the eigenbasis of the circuit observables.
            measure_all (bool): whether to perform a computational basis measurement on all qubits
                or just those specified in the script
            precision (int): decimal digits to display for parameters

        Returns:
            str: OpenQASM serialization of the circuit
        """
        wires = wires or self.wires
        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        if self.num_wires == 0:
            return qasm_str
        qasm_str += f'qreg q[{len(wires)}];\n'
        qasm_str += f'creg c[{len(wires)}];\n'
        operations = qml.transforms.convert_to_numpy_parameters(self).operations
        if rotations:
            operations += self.diagonalizing_gates
        just_ops = QuantumScript(operations)
        operations = just_ops.expand(depth=10, stop_at=lambda obj: obj.name in OPENQASM_GATES).operations
        for op in operations:
            try:
                gate = OPENQASM_GATES[op.name]
            except KeyError as e:
                raise ValueError(f'Operation {op.name} not supported by the QASM serializer') from e
            wire_labels = ','.join([f'q[{wires.index(w)}]' for w in op.wires.tolist()])
            params = ''
            if op.num_params > 0:
                if precision is not None:
                    params = '(' + ','.join([f'{p:.{precision}}' for p in op.parameters]) + ')'
                else:
                    params = '(' + ','.join([str(p) for p in op.parameters]) + ')'
            qasm_str += f'{gate}{params} {wire_labels};\n'
        if measure_all:
            for wire in range(len(wires)):
                qasm_str += f'measure q[{wire}] -> c[{wire}];\n'
        else:
            measured_wires = Wires.all_wires([m.wires for m in self.measurements])
            for w in measured_wires:
                wire_indx = self.wires.index(w)
                qasm_str += f'measure q[{wire_indx}] -> c[{wire_indx}];\n'
        return qasm_str

    @classmethod
    def from_queue(cls, queue, shots: Optional[Union[int, Sequence, Shots]]=None):
        """Construct a QuantumScript from an AnnotatedQueue."""
        return cls(*process_queue(queue), shots=shots)

    def map_to_standard_wires(self):
        """
        Map a circuit's wires such that they are in a standard order. If no
        mapping is required, the unmodified circuit is returned.

        Returns:
            QuantumScript: The circuit with wires in the standard order

        The standard order is defined by the operator wires being increasing
        integers starting at zero, to match array indices. If there are any
        measurement wires that are not in any operations, those will be mapped
        to higher values.

        **Example:**

        >>> circuit = qml.tape.QuantumScript([qml.X("a")], [qml.expval(qml.Z("b"))])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), expval(Z(1))]

        If any measured wires are not in any operations, they will be mapped last:

        >>> circuit = qml.tape.QuantumScript([qml.X(1)], [qml.probs(wires=[0, 1])])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), probs(wires=[1, 0])]

        If no wire-mapping is needed, then the returned circuit *is* the inputted circuit:

        >>> circuit = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.Z(1))])
        >>> circuit.map_to_standard_wires() is circuit
        True

        """
        op_wires = Wires.all_wires((op.wires for op in self.operations))
        meas_wires = Wires.all_wires((mp.wires for mp in self.measurements))
        num_op_wires = len(op_wires)
        meas_only_wires = set(meas_wires) - set(op_wires)
        if set(op_wires) == set(range(num_op_wires)) and meas_only_wires == set(range(num_op_wires, num_op_wires + len(meas_only_wires))):
            return self
        wire_map = {w: i for i, w in enumerate(op_wires + meas_only_wires)}
        tapes, fn = qml.map_wires(self, wire_map)
        return fn(tapes)