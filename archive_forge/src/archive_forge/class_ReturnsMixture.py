from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
class ReturnsMixture:

    def _mixture_(self) -> Iterable[Tuple[float, np.ndarray]]:
        return m