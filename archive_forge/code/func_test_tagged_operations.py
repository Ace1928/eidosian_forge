import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_tagged_operations():
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity(cirq.X(q).with_tags('tag1')) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.T(q).with_tags('tage1')) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(cirq.Ry(rads=0.1)(q).with_tags('tag1', 'tag2')) == cirq_ft.TComplexity(rotations=1)