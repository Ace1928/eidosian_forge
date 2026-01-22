from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
Tests whether a value's `_qid_shape_` and `_num_qubits_` are correct and
    consistent.

    Verifies that the entries in the shape are all positive integers and the
    length of shape equals `_num_qubits_` (and also equals `len(qubits)` if
    `val` has `qubits`.

    Args:
        val: The value under test. Should have `_qid_shape_` and/or
            `num_qubits_` methods. Can optionally have a `qubits` property.
    