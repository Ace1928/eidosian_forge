from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
@lru_cache()
def _qpos_tag(qubits: Union[cirq.Qid, Tuple[cirq.Qid]]):
    """Given a qubit or qubits, return a "position tag" (used for drawing).

    For multiple qubits, the tag is for the first qubit.
    """
    if isinstance(qubits, cirq.Qid):
        return _qpos_tag((qubits,))
    x = min(qubits)
    return f'q{x}'