import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
@pytest.mark.parametrize('options', [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION])
def test_prepare_characterization_for_moment_none_for_measurements(options):
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(cirq.measure(a, b, c, d))
    assert workflow.prepare_characterization_for_moment(moment, options) is None