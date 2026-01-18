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
def test_prepare_characterization_for_moment_fails_for_unsupported_gate(options):
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.CZ(a, b))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_characterization_for_moment(moment, options, gates_translator=_fsim_identity_converter)