import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
class Circuit(AbstractCircuit):
    """A mutable list of groups of operations to apply to some qubits.

    Methods returning information about the circuit (inherited from
    AbstractCircuit):

    *   next_moment_operating_on
    *   earliest_available_moment
    *   prev_moment_operating_on
    *   next_moments_operating_on
    *   operation_at
    *   all_qubits
    *   all_operations
    *   findall_operations
    *   findall_operations_between
    *   findall_operations_until_blocked
    *   findall_operations_with_gate_type
    *   reachable_frontier_from
    *   has_measurements
    *   are_all_matches_terminal
    *   are_all_measurements_terminal
    *   unitary
    *   final_state_vector
    *   to_text_diagram
    *   to_text_diagram_drawer
    *   qid_shape
    *   all_measurement_key_names
    *   to_quil
    *   to_qasm
    *   save_qasm
    *   get_independent_qubit_sets

    Methods for mutation:

    *   insert
    *   append
    *   insert_into_range
    *   clear_operations_touching
    *   batch_insert
    *   batch_remove
    *   batch_insert_into
    *   insert_at_frontier

    Circuits can also be iterated over,

    ```
        for moment in circuit:
            ...
    ```

    and sliced,

    *   `circuit[1:3]` is a new Circuit made up of two moments, the first being
            `circuit[1]` and the second being `circuit[2]`;
    *   `circuit[:, qubit]` is a new Circuit with the same moments, but with
            only those operations which act on the given Qubit;
    *   `circuit[:, qubits]`, where 'qubits' is list of Qubits, is a new Circuit
            with the same moments, but only with those operations which touch
            any of the given qubits;
    *   `circuit[1:3, qubit]` is equivalent to `circuit[1:3][:, qubit]`;
    *   `circuit[1:3, qubits]` is equivalent to `circuit[1:3][:, qubits]`;

    and concatenated,

    *    `circuit1 + circuit2` is a new Circuit made up of the moments in
            circuit1 followed by the moments in circuit2;

    and multiplied by an integer,

    *    `circuit * k` is a new Circuit made up of the moments in circuit repeated
            k times.

    and mutated,
    *    `circuit[1:7] = [Moment(...)]`

    and factorized,
    *   `circuit.factorize()` returns a sequence of Circuits which represent
            independent 'factors' of the original Circuit.
    """

    def __init__(self, *contents: 'cirq.OP_TREE', strategy: 'cirq.InsertStrategy'=InsertStrategy.EARLIEST) -> None:
        """Initializes a circuit.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. You can also pass in operations, lists of operations,
                or generally anything meeting the `cirq.OP_TREE` contract.
                Non-moment entries will be inserted according to the specified
                insertion strategy.
            strategy: When initializing the circuit with operations and moments
                from `contents`, this determines how the operations are packed
                together. This option does not affect later insertions into the
                circuit.
        """
        self._moments: List['cirq.Moment'] = []
        self._all_qubits: Optional[FrozenSet['cirq.Qid']] = None
        self._frozen: Optional['cirq.FrozenCircuit'] = None
        self._is_measurement: Optional[bool] = None
        self._is_parameterized: Optional[bool] = None
        self._parameter_names: Optional[AbstractSet[str]] = None
        flattened_contents = tuple(ops.flatten_to_ops_or_moments(contents))
        if all((isinstance(c, Moment) for c in flattened_contents)):
            self._moments[:] = cast(Iterable[Moment], flattened_contents)
            return
        with _compat.block_overlapping_deprecation('.*'):
            if strategy == InsertStrategy.EARLIEST:
                self._load_contents_with_earliest_strategy(flattened_contents)
            else:
                self.append(flattened_contents, strategy=strategy)

    def _mutated(self) -> None:
        """Clear cached properties in response to this circuit being mutated."""
        self._all_qubits = None
        self._frozen = None
        self._is_measurement = None
        self._is_parameterized = None
        self._parameter_names = None

    @classmethod
    def _from_moments(cls, moments: Iterable['cirq.Moment']) -> 'Circuit':
        new_circuit = Circuit()
        new_circuit._moments[:] = moments
        return new_circuit

    def _load_contents_with_earliest_strategy(self, contents: 'cirq.OP_TREE'):
        """Optimized algorithm to load contents quickly.

        The default algorithm appends operations one-at-a-time, letting them
        fall back until they encounter a moment they cannot commute with. This
        is slow because it requires re-checking for conflicts at each moment.

        Here, we instead keep track of the greatest moment that contains each
        qubit, measurement key, and control key, and append the operation to
        the moment after the maximum of these. This avoids having to check each
        moment.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. You can also pass in operations, lists of operations,
                or generally anything meeting the `cirq.OP_TREE` contract.
                Non-moment entries will be inserted according to the EARLIEST
                insertion strategy.
        """
        qubit_indices: Dict['cirq.Qid', int] = {}
        mkey_indices: Dict['cirq.MeasurementKey', int] = {}
        ckey_indices: Dict['cirq.MeasurementKey', int] = {}
        op_lists_by_index: Dict[int, List['cirq.Operation']] = defaultdict(list)
        moments_by_index: Dict[int, 'cirq.Moment'] = {}
        length = 0
        for mop in ops.flatten_to_ops_or_moments(contents):
            placement_index = get_earliest_accommodating_moment_index(mop, qubit_indices, mkey_indices, ckey_indices, length)
            length = max(length, placement_index + 1)
            if isinstance(mop, Moment):
                moments_by_index[placement_index] = mop
            else:
                op_lists_by_index[placement_index].append(mop)
        for i in range(length):
            if i in moments_by_index:
                self._moments.append(moments_by_index[i].with_operations(op_lists_by_index[i]))
            else:
                self._moments.append(Moment(op_lists_by_index[i]))

    def __copy__(self) -> 'cirq.Circuit':
        return self.copy()

    def freeze(self) -> 'cirq.FrozenCircuit':
        """Gets a frozen version of this circuit.

        Repeated calls to `.freeze()` will return the same FrozenCircuit
        instance as long as this circuit is not mutated.
        """
        from cirq.circuits.frozen_circuit import FrozenCircuit
        if self._frozen is None:
            self._frozen = FrozenCircuit.from_moments(*self._moments)
        return self._frozen

    def unfreeze(self, copy: bool=True) -> 'cirq.Circuit':
        return self.copy() if copy else self

    def all_qubits(self) -> FrozenSet['cirq.Qid']:
        if self._all_qubits is None:
            self._all_qubits = super().all_qubits()
        return self._all_qubits

    def _is_measurement_(self) -> bool:
        if self._is_measurement is None:
            self._is_measurement = super()._is_measurement_()
        return self._is_measurement

    def _is_parameterized_(self) -> bool:
        if self._is_parameterized is None:
            self._is_parameterized = super()._is_parameterized_()
        return self._is_parameterized

    def _parameter_names_(self) -> AbstractSet[str]:
        if self._parameter_names is None:
            self._parameter_names = super()._parameter_names_()
        return self._parameter_names

    def copy(self) -> 'Circuit':
        """Return a copy of this circuit."""
        copied_circuit = Circuit()
        copied_circuit._moments = self._moments[:]
        return copied_circuit

    @overload
    def __setitem__(self, key: int, value: 'cirq.Moment'):
        pass

    @overload
    def __setitem__(self, key: slice, value: Iterable['cirq.Moment']):
        pass

    def __setitem__(self, key, value):
        if isinstance(key, int) and (not isinstance(value, Moment)):
            raise TypeError('Can only assign Moments into Circuits.')
        if isinstance(key, slice):
            value = list(value)
            if any((not isinstance(v, Moment) for v in value)):
                raise TypeError('Can only assign Moments into Circuits.')
        self._moments[key] = value
        self._mutated()

    def __delitem__(self, key: Union[int, slice]):
        del self._moments[key]
        self._mutated()

    def __iadd__(self, other):
        self.append(other)
        return self

    def __add__(self, other):
        if not isinstance(other, (ops.Operation, Iterable)):
            return NotImplemented
        result = self.copy()
        return result.__iadd__(other)

    def __radd__(self, other):
        if not isinstance(other, (ops.Operation, Iterable)):
            return NotImplemented
        result = self.copy()
        result._moments[:0] = Circuit(other)._moments
        return result
    __array_priority__ = 10000

    def __imul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        self._moments *= int(repetitions)
        self._mutated()
        return self

    def __mul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        return Circuit(self._moments * int(repetitions))

    def __rmul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        return self * int(repetitions)

    def __pow__(self, exponent: int) -> 'cirq.Circuit':
        """A circuit raised to a power, only valid for exponent -1, the inverse.

        This will fail if anything other than -1 is passed to the Circuit by
        returning NotImplemented.  Otherwise this will return the inverse
        circuit, which is the circuit with its moment order reversed and for
        every moment all the moment's operations are replaced by its inverse.
        If any of the operations do not support inverse, NotImplemented will be
        returned.
        """
        if exponent != -1:
            return NotImplemented
        inv_moments = []
        for moment in self[::-1]:
            inv_moment = cirq.inverse(moment, default=NotImplemented)
            if inv_moment is NotImplemented:
                return NotImplemented
            inv_moments.append(inv_moment)
        return cirq.Circuit(inv_moments)
    __hash__ = None

    def concat_ragged(*circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str]=Alignment.LEFT) -> 'cirq.Circuit':
        return AbstractCircuit.concat_ragged(*circuits, align=align).unfreeze(copy=False)
    concat_ragged.__doc__ = AbstractCircuit.concat_ragged.__doc__

    def zip(*circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str]=Alignment.LEFT) -> 'cirq.Circuit':
        return AbstractCircuit.zip(*circuits, align=align).unfreeze(copy=False)
    zip.__doc__ = AbstractCircuit.zip.__doc__

    def transform_qubits(self, qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]) -> 'cirq.Circuit':
        """Returns the same circuit, but with different qubits.

        Args:
            qubit_map: A function or a dict mapping each current qubit into a desired
                new qubit.

        Returns:
            The receiving circuit but with qubits transformed by the given
                function.

        Raises:
            TypeError: If `qubit_function` is not a function or a dict.
        """
        if callable(qubit_map):
            transform = qubit_map
        elif isinstance(qubit_map, dict):
            transform = lambda q: qubit_map.get(q, q)
        else:
            raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')
        op_list = [Moment((operation.transform_qubits(transform) for operation in moment.operations)) for moment in self._moments]
        return Circuit(op_list)

    def earliest_available_moment(self, op: 'cirq.Operation', *, end_moment_index: Optional[int]=None) -> int:
        """Finds the index of the earliest (i.e. left most) moment which can accommodate `op`.

        Note that, unlike `circuit.prev_moment_operating_on`, this method also takes care of
        implicit dependencies between measurements and classically controlled operations (CCO)
        that depend on the results of those measurements. Therefore, using this method, a CCO
        `op` would not be allowed to move left past a measurement it depends upon.

        Args:
            op: Operation for which the earliest moment that can accommodate it needs to be found.
            end_moment_index: The moment index just after the starting point of the reverse search.
                Defaults to the length of the list of moments.

        Returns:
            Index of the earliest matching moment. Returns `end_moment_index` if no moment on left
            is available.
        """
        if end_moment_index is None:
            end_moment_index = len(self.moments)
        last_available = end_moment_index
        k = end_moment_index
        op_control_keys = protocols.control_keys(op)
        op_measurement_keys = protocols.measurement_key_objs(op)
        op_qubits = op.qubits
        while k > 0:
            k -= 1
            moment = self._moments[k]
            if moment.operates_on(op_qubits):
                return last_available
            moment_measurement_keys = moment._measurement_key_objs_()
            if not op_measurement_keys.isdisjoint(moment_measurement_keys) or not op_control_keys.isdisjoint(moment_measurement_keys) or (not moment._control_keys_().isdisjoint(op_measurement_keys)):
                return last_available
            if self._can_add_op_at(k, op):
                last_available = k
        return last_available

    def _pick_or_create_inserted_op_moment_index(self, splitter_index: int, op: 'cirq.Operation', strategy: 'cirq.InsertStrategy') -> int:
        """Determines and prepares where an insertion will occur.

        Args:
            splitter_index: The index to insert at.
            op: The operation that will be inserted.
            strategy: The insertion strategy.

        Returns:
            The index of the (possibly new) moment where the insertion should
                occur.

        Raises:
            ValueError: Unrecognized append strategy.
        """
        if strategy is InsertStrategy.NEW or strategy is InsertStrategy.NEW_THEN_INLINE:
            self._moments.insert(splitter_index, Moment())
            self._mutated()
            return splitter_index
        if strategy is InsertStrategy.INLINE:
            if 0 <= splitter_index - 1 < len(self._moments) and self._can_add_op_at(splitter_index - 1, op):
                return splitter_index - 1
            return self._pick_or_create_inserted_op_moment_index(splitter_index, op, InsertStrategy.NEW)
        if strategy is InsertStrategy.EARLIEST:
            if self._can_add_op_at(splitter_index, op):
                return self.earliest_available_moment(op, end_moment_index=splitter_index)
            return self._pick_or_create_inserted_op_moment_index(splitter_index, op, InsertStrategy.INLINE)
        raise ValueError(f'Unrecognized append strategy: {strategy}')

    def _can_add_op_at(self, moment_index: int, operation: 'cirq.Operation') -> bool:
        if not 0 <= moment_index < len(self._moments):
            return True
        return not self._moments[moment_index].operates_on(operation.qubits)

    def insert(self, index: int, moment_or_operation_tree: Union['cirq.Operation', 'cirq.OP_TREE'], strategy: 'cirq.InsertStrategy'=InsertStrategy.EARLIEST) -> int:
        """Inserts operations into the circuit.

        Operations are inserted into the moment specified by the index and
        'InsertStrategy'.
        Moments within the operation tree are inserted intact.

        Args:
            index: The index to insert all the operations at.
            moment_or_operation_tree: The moment or operation tree to insert.
            strategy: How to pick/create the moment to put operations into.

        Returns:
            The insertion index that will place operations just after the
            operations that were inserted by this method.

        Raises:
            ValueError: Bad insertion strategy.
        """
        k = max(min(index if index >= 0 else len(self._moments) + index, len(self._moments)), 0)
        for moment_or_op in list(ops.flatten_to_ops_or_moments(moment_or_operation_tree)):
            if isinstance(moment_or_op, Moment):
                self._moments.insert(k, moment_or_op)
                k += 1
            else:
                op = moment_or_op
                p = self._pick_or_create_inserted_op_moment_index(k, op, strategy)
                while p >= len(self._moments):
                    self._moments.append(Moment())
                self._moments[p] = self._moments[p].with_operation(op)
                k = max(k, p + 1)
                if strategy is InsertStrategy.NEW_THEN_INLINE:
                    strategy = InsertStrategy.INLINE
        self._mutated()
        return k

    def insert_into_range(self, operations: 'cirq.OP_TREE', start: int, end: int) -> int:
        """Writes operations inline into an area of the circuit.

        Args:
            start: The start of the range (inclusive) to write the
                given operations into.
            end: The end of the range (exclusive) to write the given
                operations into. If there are still operations remaining,
                new moments are created to fit them.
            operations: An operation or tree of operations to insert.

        Returns:
            An insertion index that will place operations after the operations
            that were inserted by this method.

        Raises:
            IndexError: Bad inline_start and/or inline_end.
        """
        if not 0 <= start <= end <= len(self):
            raise IndexError(f'Bad insert indices: [{start}, {end})')
        flat_ops = list(ops.flatten_to_ops(operations))
        i = start
        op_index = 0
        while op_index < len(flat_ops):
            op = flat_ops[op_index]
            while i < end and self._moments[i].operates_on(op.qubits):
                i += 1
            if i >= end:
                break
            self._moments[i] = self._moments[i].with_operation(op)
            op_index += 1
        self._mutated()
        if op_index >= len(flat_ops):
            return end
        return self.insert(end, flat_ops[op_index:])

    def _push_frontier(self, early_frontier: Dict['cirq.Qid', int], late_frontier: Dict['cirq.Qid', int], update_qubits: Optional[Iterable['cirq.Qid']]=None) -> Tuple[int, int]:
        """Inserts moments to separate two frontiers.

        After insertion n_new moments, the following holds:
           for q in late_frontier:
               early_frontier[q] <= late_frontier[q] + n_new
           for q in update_qubits:
               early_frontier[q] the identifies the same moment as before
                   (but whose index may have changed if this moment is after
                   those inserted).

        Args:
            early_frontier: The earlier frontier. For qubits not in the later
                frontier, this is updated to account for the newly inserted
                moments.
            late_frontier: The later frontier. This is not modified.
            update_qubits: The qubits for which to update early_frontier to
                account for the newly inserted moments.

        Returns:
            (index at which new moments were inserted, how many new moments
            were inserted) if new moments were indeed inserted. (0, 0)
            otherwise.
        """
        if update_qubits is None:
            update_qubits = set(early_frontier).difference(late_frontier)
        n_new_moments = max((early_frontier.get(q, 0) - late_frontier[q] for q in late_frontier)) if late_frontier else 0
        if n_new_moments > 0:
            insert_index = min(late_frontier.values())
            self._moments[insert_index:insert_index] = [Moment()] * n_new_moments
            self._mutated()
            for q in update_qubits:
                if early_frontier.get(q, 0) > insert_index:
                    early_frontier[q] += n_new_moments
            return (insert_index, n_new_moments)
        return (0, 0)

    def _insert_operations(self, operations: Sequence['cirq.Operation'], insertion_indices: Sequence[int]) -> None:
        """Inserts operations at the specified moments. Appends new moments if
        necessary.

        Args:
            operations: The operations to insert.
            insertion_indices: Where to insert them, i.e. operations[i] is
                inserted into moments[insertion_indices[i].

        Raises:
            ValueError: operations and insert_indices have different lengths.

        NB: It's on the caller to ensure that the operations won't conflict
        with operations already in the moment or even each other.
        """
        if len(operations) != len(insertion_indices):
            raise ValueError('operations and insertion_indices must have the same length.')
        self._moments += [Moment() for _ in range(1 + max(insertion_indices) - len(self))]
        self._mutated()
        moment_to_ops: Dict[int, List['cirq.Operation']] = defaultdict(list)
        for op_index, moment_index in enumerate(insertion_indices):
            moment_to_ops[moment_index].append(operations[op_index])
        for moment_index, new_ops in moment_to_ops.items():
            self._moments[moment_index] = self._moments[moment_index].with_operations(*new_ops)

    def insert_at_frontier(self, operations: 'cirq.OP_TREE', start: int, frontier: Optional[Dict['cirq.Qid', int]]=None) -> Dict['cirq.Qid', int]:
        """Inserts operations inline at frontier.

        Args:
            operations: The operations to insert.
            start: The moment at which to start inserting the operations.
            frontier: frontier[q] is the earliest moment in which an operation
                acting on qubit q can be placed.

        Raises:
            ValueError: If the frontier given is after start.
        """
        if frontier is None:
            frontier = defaultdict(lambda: 0)
        flat_ops = tuple(ops.flatten_to_ops(operations))
        if not flat_ops:
            return frontier
        qubits = set((q for op in flat_ops for q in op.qubits))
        if any((frontier[q] > start for q in qubits)):
            raise ValueError('The frontier for qubits on which the operationsto insert act cannot be after start.')
        next_moments = self.next_moments_operating_on(qubits, start)
        insertion_indices, _ = _pick_inserted_ops_moment_indices(flat_ops, start, frontier)
        self._push_frontier(frontier, next_moments)
        self._insert_operations(flat_ops, insertion_indices)
        return frontier

    def batch_remove(self, removals: Iterable[Tuple[int, 'cirq.Operation']]) -> None:
        """Removes several operations from a circuit.

        Args:
            removals: A sequence of (moment_index, operation) tuples indicating
                operations to delete from the moments that are present. All
                listed operations must actually be present or the edit will
                fail (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to delete wasn't present to start with.
            IndexError: Deleted from a moment that doesn't exist.
        """
        copy = self.copy()
        for i, op in removals:
            if op not in copy._moments[i].operations:
                raise ValueError(f"Can't remove {op} @ {i} because it doesn't exist.")
            copy._moments[i] = Moment((old_op for old_op in copy._moments[i].operations if op != old_op))
        self._moments = copy._moments
        self._mutated()

    def batch_replace(self, replacements: Iterable[Tuple[int, 'cirq.Operation', 'cirq.Operation']]) -> None:
        """Replaces several operations in a circuit with new operations.

        Args:
            replacements: A sequence of (moment_index, old_op, new_op) tuples
                indicating operations to be replaced in this circuit. All "old"
                operations must actually be present or the edit will fail
                (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to replace wasn't present to start with.
            IndexError: Replaced in a moment that doesn't exist.
        """
        copy = self.copy()
        for i, op, new_op in replacements:
            if op not in copy._moments[i].operations:
                raise ValueError(f"Can't replace {op} @ {i} because it doesn't exist.")
            copy._moments[i] = Moment((old_op if old_op != op else new_op for old_op in copy._moments[i].operations))
        self._moments = copy._moments
        self._mutated()

    def batch_insert_into(self, insert_intos: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
        """Inserts operations into empty spaces in existing moments.

        If any of the insertions fails (due to colliding with an existing
        operation), this method fails without making any changes to the circuit.

        Args:
            insert_intos: A sequence of (moment_index, new_op_tree)
                pairs indicating a moment to add new operations into.

        Raises:
            ValueError: One of the insertions collided with an existing
                operation.
            IndexError: Inserted into a moment index that doesn't exist.
        """
        copy = self.copy()
        for i, insertions in insert_intos:
            copy._moments[i] = copy._moments[i].with_operations(insertions)
        self._moments = copy._moments
        self._mutated()

    def batch_insert(self, insertions: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
        """Applies a batched insert operation to the circuit.

        Transparently handles the fact that earlier insertions may shift
        the index that later insertions should occur at. For example, if you
        insert an operation at index 2 and at index 4, but the insert at index 2
        causes a new moment to be created, then the insert at "4" will actually
        occur at index 5 to account for the shift from the new moment.

        All insertions are done with the strategy `cirq.InsertStrategy.EARLIEST`.

        When multiple inserts occur at the same index, the gates from the later
        inserts end up before the gates from the earlier inserts (exactly as if
        you'd called list.insert several times with the same index: the later
        inserts shift the earliest inserts forward).

        Args:
            insertions: A sequence of (insert_index, operations) pairs
                indicating operations to add into the circuit at specific
                places.
        """
        copy = self.copy()
        shift = 0
        insertions = sorted(insertions, key=lambda e: e[0])
        groups = _group_until_different(insertions, key=lambda e: e[0], val=lambda e: e[1])
        for i, group in groups:
            insert_index = i + shift
            next_index = copy.insert(insert_index, reversed(group), InsertStrategy.EARLIEST)
            if next_index > insert_index:
                shift += next_index - insert_index
        self._moments = copy._moments
        self._mutated()

    def append(self, moment_or_operation_tree: Union['cirq.Moment', 'cirq.OP_TREE'], strategy: 'cirq.InsertStrategy'=InsertStrategy.EARLIEST) -> None:
        """Appends operations onto the end of the circuit.

        Moments within the operation tree are appended intact.

        Args:
            moment_or_operation_tree: The moment or operation tree to append.
            strategy: How to pick/create the moment to put operations into.
        """
        self.insert(len(self._moments), moment_or_operation_tree, strategy)

    def clear_operations_touching(self, qubits: Iterable['cirq.Qid'], moment_indices: Iterable[int]):
        """Clears operations that are touching given qubits at given moments.

        Args:
            qubits: The qubits to check for operations on.
            moment_indices: The indices of moments to check for operations
                within.
        """
        qubits = frozenset(qubits)
        for k in moment_indices:
            if 0 <= k < len(self._moments):
                self._moments[k] = self._moments[k].without_operations_touching(qubits)
        self._mutated()

    @property
    def moments(self) -> Sequence['cirq.Moment']:
        return self._moments

    def with_noise(self, noise: 'cirq.NOISE_MODEL_LIKE') -> 'cirq.Circuit':
        """Make a noisy version of the circuit.

        Args:
            noise: The noise model to use.  This describes the kind of noise to
                add to the circuit.

        Returns:
            A new circuit with the same moment structure but with new moments
            inserted where needed when more than one noisy operation is
            generated for an input operation.  Emptied moments are removed.
        """
        noise_model = devices.NoiseModel.from_noise_model_like(noise)
        qubits = sorted(self.all_qubits())
        c_noisy = Circuit()
        for op_tree in noise_model.noisy_moments(self, qubits):
            c_noisy += Circuit(op_tree)
        return c_noisy