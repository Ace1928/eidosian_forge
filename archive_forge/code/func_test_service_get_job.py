import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
def test_service_get_job():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    job_dict = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = job_dict
    service._client = mock_client
    job = service.get_job('job_id')
    assert job.job_id() == 'job_id'
    mock_client.get_job.assert_called_with(job_id='job_id')