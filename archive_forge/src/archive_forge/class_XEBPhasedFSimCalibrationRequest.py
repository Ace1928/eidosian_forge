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
class XEBPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request for cross entropy benchmarking (XEB) calibration.

    Attributes:
        options: XEB-specific characterization options.
    """
    options: XEBPhasedFSimCalibrationOptions

    def to_calibration_layer(self) -> CalibrationLayer:
        circuit = cirq.Circuit((self.gate.on(*pair) for pair in self.pairs))
        return CalibrationLayer(calibration_type=_XEB_PHASED_FSIM_HANDLER_NAME, program=circuit, args=self.options.to_args())

    def parse_result(self, result: CalibrationResult, job: Optional[EngineJob]=None) -> PhasedFSimCalibrationResult:
        if result.code != v2.calibration_pb2.SUCCESS:
            raise PhasedFSimCalibrationError(result.error_message)
        initial_fids = _parse_xeb_fidelities_df(result.metrics, 'initial_fidelities')
        final_fids = _parse_xeb_fidelities_df(result.metrics, 'final_fidelities')
        final_params = {pair: PhasedFSimCharacterization(**angles) for pair, angles in _parse_characterized_angles(result.metrics, 'characterized_angles').items()}
        return PhasedFSimCalibrationResult(parameters=final_params, gate=self.gate, options=self.options, project_id=None if job is None else job.project_id, program_id=None if job is None else job.program_id, job_id=None if job is None else job.job_id)

    @classmethod
    def _from_json_dict_(cls, gate: cirq.Gate, pairs: List[Tuple[cirq.Qid, cirq.Qid]], options: XEBPhasedFSimCalibrationOptions, **kwargs) -> 'XEBPhasedFSimCalibrationRequest':
        instantiation_pairs = tuple(((q_a, q_b) for q_a, q_b in pairs))
        return cls(instantiation_pairs, gate, options)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)