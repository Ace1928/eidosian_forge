import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@pytest.mark.parametrize('target,expected_results', [('qpu', [[0], [1], [1], [1]]), ('simulator', [[1], [0], [1], [1]])])
def test_service_run(target, expected_results):
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = {'id': 'job_id', 'status': 'completed', 'target': target, 'metadata': {'shots': '4', 'measurement0': f'a{chr(31)}0'}, 'qubits': '1', 'status': 'completed'}
    mock_client.get_results.return_value = {'0': '0.25', '1': '0.75'}
    service._client = mock_client
    a = sympy.Symbol('a')
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X ** a)(q), cirq.measure(q, key='a'))
    params = cirq.ParamResolver({'a': 0.5})
    result = service.run(circuit=circuit, repetitions=4, target=target, name='bacon', param_resolver=params, seed=2)
    assert result == cirq.ResultDict(params=params, measurements={'a': np.array(expected_results)})
    create_job_kwargs = mock_client.create_job.call_args[1]
    assert create_job_kwargs['serialized_program'].body['qubits'] == 1
    assert create_job_kwargs['serialized_program'].metadata == {'measurement0': f'a{chr(31)}0'}
    assert create_job_kwargs['repetitions'] == 4
    assert create_job_kwargs['target'] == target
    assert create_job_kwargs['name'] == 'bacon'