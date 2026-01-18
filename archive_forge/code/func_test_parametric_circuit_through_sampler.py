from typing import cast, Tuple
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler, circuit_sweep_executors
@pytest.mark.rigetti_integration
def test_parametric_circuit_through_sampler(parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]) -> None:
    """test that RigettiQCSSampler can run a basic parametric circuit on the
    QVM and return an accurate list of `cirq.study.Result`.
    """
    circuit, sweepable = parametric_circuit_with_params
    qc = get_qc('9q-square', as_qvm=True)
    sampler = RigettiQCSSampler(quantum_computer=qc)
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0
    repetitions = 10
    results = sampler.run_sweep(program=circuit, params=sweepable, repetitions=repetitions)
    assert len(sweepable) == len(results)
    expected_results = [(10, 0), (4, 6), (0, 10), (4, 6), (10, 0)]
    for i, result in enumerate(results):
        assert isinstance(result, cirq.study.Result)
        assert sweepable[i] == result.params
        assert 'm' in result.measurements
        assert (repetitions, 1) == result.measurements['m'].shape
        counter = result.histogram(key='m')
        assert expected_results[i][0] == counter[0]
        assert expected_results[i][1] == counter[1]