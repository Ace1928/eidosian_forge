from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_with_random_gaussian_sqrt_iswap_fails_with_invalid_mean():
    with pytest.raises(ValueError):
        PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(mean=PhasedFSimCharacterization(theta=np.pi / 4))