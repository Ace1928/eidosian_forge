import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {'IONQ_API_KEY': 'tomyheart'})
def test_service_api_key_from_env_var_ionq():
    service = ionq.Service(remote_host='http://example.com')
    assert service.api_key == 'tomyheart'