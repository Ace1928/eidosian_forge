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
def to_zphase_data(results: Iterable[PhasedFSimCalibrationResult]) -> util.ZPhaseDataType:
    """Packages a collection of results into ZPhaseDataType.

    Args:
        results: List of results to pack into ZPhaseDataType. If multiple results provide a value
            for a given (gate, angle, qubits) tuple, only the last one will be kept.

    Returns:
        A ZPhaseDataType-formatted result representation. This can be used with the
            calibration-to-noise pipeline for generating noise models.

    Raises:
        ValueError: if results for a gate other than Sycamore or ISwapPowGate are given.
    """
    zphase_data: util.ZPhaseDataType = {}
    for result in results:
        gate_type = GATE_ZPHASE_CODE_PAIRS.get(type(result.gate))
        if gate_type is None:
            raise ValueError(f"Only 'SycamoreGate' and 'ISwapPowGate' are supported, got {result.gate}")
        gate_dict = zphase_data.setdefault(gate_type, {})
        for qubits, data in result.parameters.items():
            for angle, value in data.asdict().items():
                if value is None:
                    continue
                angle_dict = gate_dict.setdefault(angle, {})
                angle_dict[qubits] = value
    return zphase_data