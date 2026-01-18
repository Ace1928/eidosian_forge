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
def test_prepare_characterization_for_operations(options_cls):
    options, cls = options_cls
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    circuit_1 = cirq.Circuit([[cirq.X(q00), cirq.Y(q11)], [SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)], [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(q01)]])
    requests_1 = workflow.prepare_characterization_for_operations(circuit_1, options=options)
    assert requests_1 == [cls(pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options)]
    circuit_2 = cirq.Circuit([[SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)], [SQRT_ISWAP_INV_GATE.on(q00, q10), SQRT_ISWAP_INV_GATE.on(q01, q11)], [SQRT_ISWAP_INV_GATE.on(q10, q20), SQRT_ISWAP_INV_GATE.on(q11, q21)]])
    requests_2 = workflow.prepare_characterization_for_operations([circuit_1, circuit_2], options=options)
    assert requests_2 == [cls(pairs=((q00, q10), (q11, q21)), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((q01, q11), (q10, q20)), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options), cls(pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options)]