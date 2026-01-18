import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def try_convert_sqrt_iswap_to_fsim(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    """Converts an equivalent gate to FSimGate(theta=π/4, phi=0) if possible.

    Args:
        gate: Gate to verify.

    Returns:
        FSimGateCalibration with engine_gate FSimGate(theta=π/4, phi=0) if the provided gate is
        either FSimGate, ISWapPowGate, PhasedFSimGate or PhasedISwapPowGate that is equivalent to
        FSimGate(theta=±π/4, phi=0). None otherwise.
    """
    result = try_convert_gate_to_fsim(gate)
    if result is None:
        return None
    engine_gate = result.engine_gate
    if math.isclose(cast(float, engine_gate.theta), np.pi / 4) and math.isclose(cast(float, engine_gate.phi), 0.0):
        return result
    return None