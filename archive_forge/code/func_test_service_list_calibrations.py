import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
def test_service_list_calibrations():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    calibrations = [{'id': '1', 'qubits': '1'}, {'id': '2', 'qubits': 2}]
    mock_client.list_calibrations.return_value = calibrations
    service._client = mock_client
    start = datetime.datetime.utcfromtimestamp(1284286794)
    end = datetime.datetime.utcfromtimestamp(1284286795)
    listed_calibrations = service.list_calibrations(start=start, end=end, limit=10, batch_size=2)
    assert listed_calibrations[0].num_qubits() == 1
    assert listed_calibrations[1].num_qubits() == 2
    mock_client.list_calibrations.assert_called_with(start=start, end=end, limit=10, batch_size=2)