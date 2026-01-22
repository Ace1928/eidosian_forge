from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def _generate_measurement_ids(self) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    meas_key_id_map: Dict[str, str] = {}
    meas_comments: Dict[str, Optional[str]] = {}
    meas_i = 0
    for meas in self.measurements:
        key = protocols.measurement_key_name(meas)
        if key in meas_key_id_map:
            continue
        meas_id = f'm_{key}'
        if self.is_valid_qasm_id(meas_id):
            meas_comments[key] = None
        else:
            meas_id = f'm{meas_i}'
            meas_i += 1
            meas_comments[key] = ' '.join(key.split('\n'))
        meas_key_id_map[key] = meas_id
    return (meas_key_id_map, meas_comments)