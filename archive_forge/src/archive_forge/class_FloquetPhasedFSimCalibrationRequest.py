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
class FloquetPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request specific to Floquet calibration.

    Attributes:
        options: Floquet-specific characterization options.
    """
    options: FloquetPhasedFSimCalibrationOptions

    @classmethod
    def from_moment(cls, moment: cirq.Moment, options: FloquetPhasedFSimCalibrationOptions):
        """Creates a FloquetPhasedFSimCalibrationRequest from a Moment.

        Given a `Moment` object, this function extracts out the pairs of
        qubits and the `Gate` used to create a `FloquetPhasedFSimCalibrationRequest`
        object.  The moment must contain only identical two-qubit FSimGates.
        If dissimilar gates are passed in, a ValueError is raised.
        """
        pairs, gate = _create_pairs_from_moment(moment)
        return cls(pairs, gate, options)

    def to_calibration_layer(self) -> CalibrationLayer:
        circuit = cirq.Circuit((self.gate.on(*pair) for pair in self.pairs))
        if self.options.measure_qubits is not None:
            circuit += cirq.Moment(cirq.measure(*self.options.measure_qubits))
        args: Dict[str, Any] = {'est_theta': self.options.characterize_theta, 'est_zeta': self.options.characterize_zeta, 'est_chi': self.options.characterize_chi, 'est_gamma': self.options.characterize_gamma, 'est_phi': self.options.characterize_phi, 'readout_corrections': True, 'version': self.options.version}
        if self.options.readout_error_tolerance is not None:
            args['readout_error_tolerance'] = self.options.readout_error_tolerance
            args['correlated_readout_error_tolerance'] = _correlated_from_readout_tolerance(self.options.readout_error_tolerance)
        return CalibrationLayer(calibration_type=_FLOQUET_PHASED_FSIM_HANDLER_NAME, program=circuit, args=args)

    def parse_result(self, result: CalibrationResult, job: Optional[EngineJob]=None) -> PhasedFSimCalibrationResult:
        if result.code != v2.calibration_pb2.SUCCESS:
            raise PhasedFSimCalibrationError(result.error_message)
        decoded: Dict[int, Dict[str, Any]] = collections.defaultdict(lambda: {})
        for keys, values in result.metrics['angles'].items():
            for key, value in zip(keys, values):
                match = re.match('(\\d+)_(.+)', str(key))
                if not match:
                    raise ValueError(f'Unknown metric name {key}')
                index = int(match[1])
                name = match[2]
                decoded[index][name] = value
        parsed = {}
        for data in decoded.values():
            a = v2.qubit_from_proto_id(data['qubit_a'])
            b = v2.qubit_from_proto_id(data['qubit_b'])
            parsed[a, b] = PhasedFSimCharacterization(theta=data.get('theta_est', None), zeta=data.get('zeta_est', None), chi=data.get('chi_est', None), gamma=data.get('gamma_est', None), phi=data.get('phi_est', None))
        return PhasedFSimCalibrationResult(parameters=parsed, gate=self.gate, options=self.options, project_id=None if job is None else job.project_id, program_id=None if job is None else job.program_id, job_id=None if job is None else job.job_id)

    @classmethod
    def _from_json_dict_(cls, gate: cirq.Gate, pairs: List[Tuple[cirq.Qid, cirq.Qid]], options: FloquetPhasedFSimCalibrationOptions, **kwargs) -> 'FloquetPhasedFSimCalibrationRequest':
        """Magic method for the JSON serialization protocol.

        Converts serialized dictionary into a dict suitable for
        class instantiation."""
        instantiation_pairs = tuple(((q_a, q_b) for q_a, q_b in pairs))
        return cls(instantiation_pairs, gate, options)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {'pairs': [(pair[0], pair[1]) for pair in self.pairs], 'gate': self.gate, 'options': self.options}