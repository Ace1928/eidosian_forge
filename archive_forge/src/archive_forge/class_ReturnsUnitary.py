from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
class ReturnsUnitary:

    def _unitary_(self) -> np.ndarray:
        return u