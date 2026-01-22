import collections
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
import pandas as pd
from cirq import ops, protocols, study, value
from cirq.work.observable_measurement import (
from cirq.work.observable_settings import _hashable_param
@staticmethod
def _get_measurement_shapes(circuit: 'cirq.AbstractCircuit') -> Dict[str, Tuple[int, Tuple[int, ...]]]:
    """Gets the shapes of measurements in the given circuit.

        Returns:
            A mapping from measurement key name to a tuple of (num_instances, qid_shape),
            where num_instances is the number of times that key appears in the circuit and
            qid_shape is the shape of measured qubits for the key, as determined by the
            `cirq.qid_shape` protocol.

        Raises:
            ValueError: if the qid_shape of different instances of the same measurement
            key disagree.
        """
    qid_shapes: Dict[str, Tuple[int, ...]] = {}
    num_instances: Dict[str, int] = collections.Counter()
    for op in circuit.all_operations():
        key = protocols.measurement_key_name(op, default=None)
        if key is not None:
            qid_shape = protocols.qid_shape(op)
            prev_qid_shape = qid_shapes.setdefault(key, qid_shape)
            if qid_shape != prev_qid_shape:
                raise ValueError(f'Different qid shapes for repeated measurement: key={key!r}, prev_qid_shape={prev_qid_shape}, qid_shape={qid_shape}')
            num_instances[key] += 1
    return {k: (num_instances[k], qid_shape) for k, qid_shape in qid_shapes.items()}