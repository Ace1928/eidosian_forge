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
def test_prepare_floquet_characterization_for_operations_when_multiple_gates_fails():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit([SQRT_ISWAP_INV_GATE.on(q00, q01), cirq.FSimGate(theta=0.0, phi=np.pi).on(q00, q01)])
    with pytest.raises(ValueError):
        workflow.prepare_floquet_characterization_for_operations(circuit, gates_translator=_fsim_identity_converter)