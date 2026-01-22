from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _is_density_matrix(self) -> bool:
    """Whether this quantum state is a density matrix."""
    return self.data.shape == (self._dim, self._dim)