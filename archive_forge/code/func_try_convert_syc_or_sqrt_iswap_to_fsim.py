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
def try_convert_syc_or_sqrt_iswap_to_fsim(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    """Converts a gate to equivalent PhaseCalibratedFSimGate if possible.

    Args:
        gate: Gate to convert.

    Returns:
        Equivalent PhaseCalibratedFSimGate if its `engine_gate` is Sycamore or inverse sqrt(iSWAP)
        gate. Otherwise returns None.
    """
    result = try_convert_gate_to_fsim(gate)
    if result is None:
        return None
    engine_gate = result.engine_gate
    if math.isclose(engine_gate.theta, np.pi / 2) and math.isclose(engine_gate.phi, np.pi / 6):
        return result
    if math.isclose(engine_gate.theta, np.pi / 4) and math.isclose(engine_gate.phi, 0.0):
        return result
    return None