from unittest import mock
import pandas as pd
import sympy as sp
import cirq
import cirq_ionq as ionq
def test_sampler_simulator():
    mock_service = mock.MagicMock()
    job_dict = {'id': '1', 'status': 'completed', 'qubits': '1', 'target': 'simulator', 'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'}}
    job = ionq.Job(client=mock_service, job_dict=job_dict)
    mock_service.create_job.return_value = job
    mock_service.get_results.return_value = {'0': '0.25', '1': '0.75'}
    sampler = ionq.Sampler(service=mock_service, target='simulator', seed=10)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(results, pd.DataFrame(columns=['a'], index=[0, 1, 2, 3], data=[[1], [0], [1], [1]]))
    mock_service.create_job.assert_called_once_with(circuit=circuit, repetitions=4, target='simulator')