import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qroam_hashable():
    qrom = cirq_ft.SelectSwapQROM([1, 2, 5, 6, 7, 8])
    assert hash(qrom) is not None
    assert cirq_ft.t_complexity(qrom) == cirq_ft.TComplexity(32, 160, 0)