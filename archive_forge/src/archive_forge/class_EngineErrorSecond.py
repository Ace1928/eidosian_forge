from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
class EngineErrorSecond(EngineReturn):
    """A put mock class for testing error responses
    This will return an error on the second put call"""

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            self.test_dict['status'] = 'error'
        return self