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
def test_xeb_parse_bad_fidelities():
    metrics = cirq_google.Calibration(metrics={'initial_fidelities_depth_5': {('layer_0', 'pair_0', cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)): [1.0]}})
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    pd.testing.assert_frame_equal(df, pd.DataFrame({'cycle_depth': [5], 'layer_i': [0], 'pair_i': [0], 'fidelity': [1.0], 'pair': [(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1))]}))
    metrics = cirq_google.Calibration(metrics={'initial_fidelities_depth_5x': {('layer_0', 'pair_0', '0_0', '1_1'): [1.0]}})
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    assert len(df) == 0, 'bad metric name ignored'
    metrics = cirq_google.Calibration(metrics={'initial_fidelities_depth_5': {('bad_name_0', 'pair_0', '0_0', '1_1'): [1.0]}})
    with pytest.raises(ValueError, match='Could not parse layer value for bad_name_0'):
        _parse_xeb_fidelities_df(metrics, 'initial_fidelities')