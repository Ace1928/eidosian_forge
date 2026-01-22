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
@dataclasses.dataclass(frozen=True)
class LocalXEBPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request for local cross entropy benchmarking (XEB) calibration.

    A "Local" request (corresponding to `LocalXEBPhasedFSimCalibrationOptions`) instructs
    `cirq_google.run_calibrations` to execute XEB analysis locally (not via the quantum
    engine). As such, `run_calibrations` can work with any `cirq.Sampler`, not just
    `ProcessorSampler`.

    Attributes:
        options: local-XEB-specific characterization options.
    """
    options: LocalXEBPhasedFSimCalibrationOptions

    def parse_result(self, result: CalibrationResult, job: Optional[EngineJob]=None) -> PhasedFSimCalibrationResult:
        raise NotImplementedError('Not applicable for local calibrations')

    def to_calibration_layer(self) -> CalibrationLayer:
        raise NotImplementedError('Not applicable for local calibrations')

    @classmethod
    def _from_json_dict_(cls, gate: cirq.Gate, pairs: List[Tuple[cirq.Qid, cirq.Qid]], options: LocalXEBPhasedFSimCalibrationOptions, **kwargs) -> 'LocalXEBPhasedFSimCalibrationRequest':
        instantiation_pairs = tuple(((q_a, q_b) for q_a, q_b in pairs))
        return cls(instantiation_pairs, gate, options)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)