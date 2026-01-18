import datetime
import os
from unittest import mock
import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
import cirq_ionq as ionq
@mock.patch.dict(os.environ, {}, clear=True)
def test_service_remote_host_default():
    service = ionq.Service(api_key='tomyheart', api_version='v0.3')
    assert service.remote_host == 'https://api.ionq.co/v0.3'