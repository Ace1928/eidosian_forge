import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def test_xeb_parse_angles():
    q0, q1, q2, q3 = [cirq.GridQubit(0, index) for index in range(4)]
    result = _load_xeb_results_textproto()
    metrics = result.metrics
    angles = _parse_characterized_angles(metrics, 'characterized_angles')
    assert angles == {(q0, q1): {'theta': -0.7853981, 'phi': 0.0}, (q2, q3): {'theta': -0.7853981, 'phi': 0.0}}