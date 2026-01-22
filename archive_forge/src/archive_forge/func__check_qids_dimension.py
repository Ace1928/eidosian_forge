import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
def _check_qids_dimension(qids):
    """A utility to check that we only have Qubits."""
    for qid in qids:
        if qid.dimension != 2:
            raise ValueError(f'Only qubits are supported, but {qid} has dimension {qid.dimension}')