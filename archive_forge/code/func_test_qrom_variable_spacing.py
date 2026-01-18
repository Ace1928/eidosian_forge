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
def test_qrom_variable_spacing():
    data = [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
    assert cirq_ft.t_complexity(cirq_ft.QROM.build(data)).t == (8 - 2) * 4
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    assert cirq_ft.t_complexity(cirq_ft.QROM.build(data)).t == (5 - 2) * 4
    data = [1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]
    assert cirq_ft.t_complexity(cirq_ft.QROM.build(data)).t == (8 - 2) * 4
    data = [1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    assert cirq_ft.t_complexity(cirq_ft.QROM.build(data, data)).t == (5 - 2) * 4
    assert cirq_ft.t_complexity(cirq_ft.QROM.build(data, 2 * np.array(data))).t == (5 - 2) * 4
    qrom = cirq_ft.QROM.build(np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]]))
    _assert_qrom_has_diagram(qrom, '\nselection00: ───X───@───X───@───\n                    │       │\ntarget00: ──────────┼───────X───\n                    │\ntarget01: ──────────X───────────\n    ')
    qrom = cirq_ft.QROM.build(np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]], dtype=int))
    _assert_qrom_has_diagram(qrom, '\nselection00: ───X───@─────────@───────@──────X───@─────────@───────@──────\n                    │         │       │          │         │       │\nselection10: ───────(0)───────┼───────@──────────(0)───────┼───────@──────\n                    │         │       │          │         │       │\nanc_1: ─────────────And───@───X───@───And†───────And───@───X───@───And†───\n                          │       │                    │       │\ntarget00: ────────────────┼───────┼────────────────────X───────X──────────\n                          │       │\ntarget01: ────────────────X───────X───────────────────────────────────────\n        ')
    assert cirq_ft.t_complexity(cirq_ft.QROM.build([3, 3, 3, 3])).t == 0