import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_t_complexity():
    with pytest.raises(TypeError):
        _ = cirq_ft.t_complexity(DoesNotSupportTComplexity())
    with pytest.raises(TypeError):
        _ = cirq_ft.t_complexity(DoesNotSupportTComplexityGate())
    assert cirq_ft.t_complexity(DoesNotSupportTComplexity(), fail_quietly=True) is None
    assert cirq_ft.t_complexity([DoesNotSupportTComplexity()], fail_quietly=True) is None
    assert cirq_ft.t_complexity(DoesNotSupportTComplexityGate(), fail_quietly=True) is None
    assert cirq_ft.t_complexity(SupportTComplexity()) == cirq_ft.TComplexity(t=1)
    g = cirq_ft.testing.GateHelper(SupportsTComplexityGateWithRegisters())
    assert g.gate._decompose_with_context_(g.operation.qubits) is NotImplemented
    assert cirq_ft.t_complexity(g.gate) == cirq_ft.TComplexity(t=1, clifford=2)
    assert cirq_ft.t_complexity(g.operation) == cirq_ft.TComplexity(t=1, clifford=2)
    assert cirq_ft.t_complexity([cirq.T, cirq.X]) == cirq_ft.TComplexity(t=1, clifford=1)
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity([cirq.T(q), cirq.X(q)]) == cirq_ft.TComplexity(t=1, clifford=1)