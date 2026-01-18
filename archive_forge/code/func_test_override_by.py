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
def test_override_by():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3)
    other = PhasedFSimCharacterization(gamma=0.4, phi=0.5, theta=0.6)
    assert characterization.override_by(other) == PhasedFSimCharacterization(theta=0.6, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5)