import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobjExperimentConfig(QobjDictField):
    """Configuration for a single OpenQASM 2 experiment in the qobj."""

    def __init__(self, calibrations=None, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        """
        Args:
            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.
            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.
            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.
            kwargs: Additional free form key value fields to add to the configuration
        """
        if calibrations:
            self.calibrations = calibrations
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        super().__init__(**kwargs)

    def to_dict(self):
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, 'calibrations'):
            out_dict['calibrations'] = self.calibrations.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if 'calibrations' in data:
            calibrations = data.pop('calibrations')
            data['calibrations'] = QasmExperimentCalibrations.from_dict(calibrations)
        return cls(**data)