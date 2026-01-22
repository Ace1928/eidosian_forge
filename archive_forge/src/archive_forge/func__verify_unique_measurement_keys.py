import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def _verify_unique_measurement_keys(operations: Iterable[cirq.Operation]):
    seen: Set[str] = set()
    for op in operations:
        if isinstance(op.gate, cirq.MeasurementGate):
            meas = op.gate
            key = cirq.measurement_key_name(meas)
            if key in seen:
                raise ValueError(f'Measurement key {key} repeated')
            seen.add(key)