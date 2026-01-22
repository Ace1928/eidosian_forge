from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def _positions(_mi, _these_qubits):
    return _add_to_positions(positions, _mi, _these_qubits, all_qubits=qubits, x_scale=x_scale, y_scale=y_scale, x_nudge=x_nudge, yb_offset=yb_offset)