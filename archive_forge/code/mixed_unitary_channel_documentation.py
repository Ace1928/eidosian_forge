from typing import Any, Dict, FrozenSet, Iterable, Mapping, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types
Creates a copy of a mixture with the given measurement key.