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
def test_options_phase_corrected_override():
    assert ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION.zeta_chi_gamma_correction_override() == PhasedFSimCharacterization(zeta=0.0, chi=0.0, gamma=0.0)
    assert FloquetPhasedFSimCalibrationOptions(characterize_theta=False, characterize_zeta=False, characterize_chi=False, characterize_gamma=False, characterize_phi=False).zeta_chi_gamma_correction_override() == PhasedFSimCharacterization()