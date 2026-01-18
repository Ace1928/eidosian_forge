import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {'IONQ_API_KEY': 'not_this_key'})
def test_service_api_key_passed_directly():
    service = ionq.Service(remote_host='http://example.com', api_key='tomyheart')
    assert service.api_key == 'tomyheart'