import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_unary_iteration_loop_empty_range():
    qm = cirq.ops.SimpleQubitManager()
    assert list(cirq_ft.unary_iteration(4, 4, [], [], [cirq.q('s')], qm)) == []
    assert list(cirq_ft.unary_iteration(4, 3, [], [], [cirq.q('s')], qm)) == []