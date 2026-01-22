from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
class HasUnitary:

    def _unitary_(self) -> np.ndarray:
        return m