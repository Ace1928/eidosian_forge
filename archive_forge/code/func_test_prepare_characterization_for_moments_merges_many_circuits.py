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
@pytest.mark.parametrize('options_cls', [(WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, cirq_google.FloquetPhasedFSimCalibrationRequest), (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest)])
def test_prepare_characterization_for_moments_merges_many_circuits(options_cls):
    options, cls = options_cls
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit_1 = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [SQRT_ISWAP_INV_GATE.on(b, c)], [SQRT_ISWAP_INV_GATE.on(a, b)]])
    circuit_with_calibration_1, requests_1 = workflow.prepare_characterization_for_moments(circuit_1, options=options)
    assert requests_1 == [cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options)]
    assert circuit_with_calibration_1.circuit == circuit_1
    assert circuit_with_calibration_1.moment_to_calibration == [None, 0, 1, 0]
    circuit_2 = cirq.Circuit([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])
    circuit_with_calibration_2, requests_2 = workflow.prepare_characterization_for_moments(circuit_2, options=options, initial=requests_1)
    assert requests_2 == [cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options)]
    assert circuit_with_calibration_2.circuit == circuit_2
    assert circuit_with_calibration_2.moment_to_calibration == [1]