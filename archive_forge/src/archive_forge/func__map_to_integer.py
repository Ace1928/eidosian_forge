from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _map_to_integer(values, uniques):
    """Map values based on its position in uniques."""
    table = _nandict({val: i for i, val in enumerate(uniques)})
    return np.array([table[v] for v in values])