from unittest import mock
import pandas as pd
import sympy as sp
import cirq
import cirq_ionq as ionq
def test_sampler_run_sweep():
    mock_service = mock.MagicMock()
    job_dict = {'id': '1', 'status': 'completed', 'qubits': '1', 'target': 'qpu', 'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'}}
    job = ionq.Job(client=mock_service, job_dict=job_dict)
    mock_service.create_job.return_value = job
    mock_service.get_results.return_value = {'0': '0.25', '1': '0.75'}
    sampler = ionq.Sampler(service=mock_service, target='qpu')
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0) ** sp.Symbol('theta'), cirq.measure(q0, key='a'))
    sweep = cirq.Linspace(key='theta', start=0.0, stop=2.0, length=5)
    results = sampler.run_sweep(program=circuit, params=sweep, repetitions=4)
    assert len(results) == 5
    assert all([result.repetitions == 4 for result in results])
    assert mock_service.create_job.call_count == 5
    for call in mock_service.create_job.call_args_list:
        _, kwargs = call
        assert kwargs['repetitions'] == 4
        assert kwargs['target'] == 'qpu'