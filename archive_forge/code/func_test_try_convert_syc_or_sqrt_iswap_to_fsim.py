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
def test_try_convert_syc_or_sqrt_iswap_to_fsim():

    def check_converts(gate: cirq.Gate):
        result = try_convert_syc_or_sqrt_iswap_to_fsim(gate)
        assert np.allclose(cirq.unitary(gate), cirq.unitary(result))

    def check_none(gate: cirq.Gate):
        assert try_convert_syc_or_sqrt_iswap_to_fsim(gate) is None
    check_converts(cirq_google.ops.SYC)
    check_converts(cirq.FSimGate(np.pi / 2, np.pi / 6))
    check_none(cirq.FSimGate(0, np.pi))
    check_converts(cirq.FSimGate(np.pi / 4, 0.0))
    check_none(cirq.FSimGate(0.2, 0.3))
    check_converts(cirq.ISwapPowGate(exponent=0.5))
    check_converts(cirq.ISwapPowGate(exponent=-0.5))
    check_none(cirq.ISwapPowGate(exponent=0.3))
    check_converts(cirq.PhasedFSimGate(theta=np.pi / 4, phi=0.0, chi=0.7))
    check_none(cirq.PhasedFSimGate(theta=0.3, phi=0.4))
    check_converts(cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=0.75))
    check_none(cirq.PhasedISwapPowGate(exponent=0.4, phase_exponent=0.75))
    check_none(cirq.ops.CZPowGate(exponent=1.0))
    check_none(cirq.CX)