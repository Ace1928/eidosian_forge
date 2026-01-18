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
def test_xeb_parse_fidelities():
    q0, q1, q2, q3 = [cirq.GridQubit(0, index) for index in range(4)]
    pair0 = (q0, q1)
    pair1 = (q2, q3)
    result = _load_xeb_results_textproto()
    metrics = result.metrics
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    should_be = pd.DataFrame({'cycle_depth': [5, 5, 25, 25], 'layer_i': [0, 0, 0, 0], 'pair_i': [0, 1, 0, 1], 'fidelity': [0.99, 0.99, 0.88, 0.88], 'pair': [pair0, pair1, pair0, pair1]})
    pd.testing.assert_frame_equal(df, should_be)
    df = _parse_xeb_fidelities_df(metrics, 'final_fidelities')
    should_be = pd.DataFrame({'cycle_depth': [5, 5, 25, 25], 'layer_i': [0, 0, 0, 0], 'pair_i': [0, 1, 0, 1], 'fidelity': [0.99, 0.99, 0.98, 0.98], 'pair': [pair0, pair1, pair0, pair1]})
    pd.testing.assert_frame_equal(df, should_be)