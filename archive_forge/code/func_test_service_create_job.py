import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
def test_service_create_job():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = {'id': 'job_id', 'status': 'completed'}
    service._client = mock_client
    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    job = service.create_job(circuit=circuit, repetitions=100, target='qpu', name='bacon')
    assert job.status() == 'completed'
    create_job_kwargs = mock_client.create_job.call_args[1]
    assert create_job_kwargs['serialized_program'].body['qubits'] == 1
    assert create_job_kwargs['repetitions'] == 100
    assert create_job_kwargs['target'] == 'qpu'
    assert create_job_kwargs['name'] == 'bacon'