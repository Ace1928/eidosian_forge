from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
Given a circuit with mixtures or channels, contract a tensor network
    representing the resultant density matrix.

    Note: If the circuit contains 6 qubits or fewer, we use a bespoke
    contraction ordering that corresponds to the "normal" in-time contraction
    ordering. Otherwise, the contraction order determination could take
    longer than doing the contraction. Your mileage may vary and benchmarking
    is encouraged for your particular problem if performance is important.
    