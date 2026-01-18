import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qrom_repr():
    data = [np.array([1, 2]), np.array([3, 5])]
    selection_bitsizes = tuple(((s - 1).bit_length() for s in data[0].shape))
    target_bitsizes = tuple((int(np.max(d)).bit_length() for d in data))
    qrom = cirq_ft.QROM(data, selection_bitsizes, target_bitsizes)
    cirq.testing.assert_equivalent_repr(qrom, setup_code='import cirq_ft\nimport numpy as np')