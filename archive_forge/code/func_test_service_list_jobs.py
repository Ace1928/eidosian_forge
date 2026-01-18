import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
def test_service_list_jobs():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    jobs = [{'id': '1'}, {'id': '2'}]
    mock_client.list_jobs.return_value = jobs
    service._client = mock_client
    listed_jobs = service.list_jobs(status='completed', limit=10, batch_size=2)
    assert listed_jobs[0].job_id() == '1'
    assert listed_jobs[1].job_id() == '2'
    mock_client.list_jobs.assert_called_with(status='completed', limit=10, batch_size=2)