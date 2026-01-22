import itertools
from typing import TYPE_CHECKING, Type, Callable, Dict, Optional, Union, Iterable, Sequence, List
from cirq import ops, circuits, protocols, _import
from cirq.transformers import transformer_api
def _stratify_circuit(circuit: circuits.AbstractCircuit, *, context: 'cirq.TransformerContext', classifiers: Sequence[Classifier]) -> 'cirq.Circuit':
    """Performs the stratification by iterating through the operations in the
    circuit and using the given classifiers to align them.

    Tagged Operations marked with any of `context.tags_to_ignore` are treated as separate
    categories and left in their original moments without stratification.

    Args:
        circuit: The circuit to break out into homogeneous moments. Will not be edited.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        classifiers: A list of rules to align the circuit. Must be exhaustive, i.e. all operations
                    will be caught by one of the processors.

    Returns:
        The stratified circuit.
    """
    num_classes = len(classifiers) + 1
    new_moments: List[List['cirq.Operation']] = []
    qubit_time_index: Dict['cirq.Qid', int] = {}
    measurement_time_index: Dict['cirq.MeasurementKey', int] = {}
    control_time_index: Dict['cirq.MeasurementKey', int] = {}
    last_ignored_ops_time_index = 0
    for moment in circuit:
        ignored_ops = []
        op_time_indices = {}
        for op in moment:
            min_time_index_for_op = circuits.circuit.get_earliest_accommodating_moment_index(op, qubit_time_index, measurement_time_index, control_time_index)
            ignored_op = any((tag in op.tags for tag in context.tags_to_ignore))
            if not ignored_op:
                op_class = _get_op_class(op, classifiers)
            else:
                op_class = len(classifiers)
                ignored_ops.append(op)
                min_time_index_for_op = max(min_time_index_for_op, last_ignored_ops_time_index + 1)
            time_index = min_time_index_for_op // num_classes * num_classes + op_class
            if time_index < min_time_index_for_op:
                time_index += num_classes
            op_time_indices[op] = time_index
        if ignored_ops:
            last_ignored_ops_time_index = max((op_time_indices[op] for op in ignored_ops))
            for op in ignored_ops:
                op_time_indices[op] = last_ignored_ops_time_index
        for op, time_index in op_time_indices.items():
            if time_index >= len(new_moments):
                new_moments += [[] for _ in range(num_classes)]
            new_moments[time_index].append(op)
            for qubit in op.qubits:
                qubit_time_index[qubit] = time_index
            for key in protocols.measurement_key_objs(op):
                measurement_time_index[key] = time_index
            for key in protocols.control_keys(op):
                control_time_index[key] = time_index
    return circuits.Circuit((circuits.Moment(moment) for moment in new_moments if moment))