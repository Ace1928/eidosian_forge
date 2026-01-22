from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
class EngineReturn:
    """A put mock class for testing the REST interface"""

    def __init__(self):
        self.test_dict = {'status': 'queued', 'id': '2131da', 'samples': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.counter = 0

    def json(self):
        self.counter += 1
        return self.test_dict

    def update(self, *args, **kwargs):
        if self.counter >= 2:
            self.test_dict['status'] = 'finished'
        return self