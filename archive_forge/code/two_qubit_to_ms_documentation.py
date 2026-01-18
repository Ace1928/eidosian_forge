from typing import Iterable, List, Optional, cast, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions, two_qubit_to_cz
Yields non-local operation of KAK decomposition.