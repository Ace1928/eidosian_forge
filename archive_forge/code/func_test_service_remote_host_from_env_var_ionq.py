import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {'IONQ_REMOTE_HOST': 'http://example.com'})
def test_service_remote_host_from_env_var_ionq():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'