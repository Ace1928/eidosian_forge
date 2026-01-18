from typing import Tuple
from numpy.typing import NDArray
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_programmable_rotation_gate_array_consistent():
    with pytest.raises(ValueError, match='must be of same length'):
        _ = CustomProgrammableRotationGateArray([1, 2], [1], kappa=1, rotation_gate=cirq.X)