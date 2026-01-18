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
def test_prepare_floquet_characterization_for_moments_merges_sub_sets():
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [SQRT_ISWAP_INV_GATE.on(b, c)], [SQRT_ISWAP_INV_GATE.on(a, b)]])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(circuit, options=options)
    assert requests == [cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options), cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options)]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]