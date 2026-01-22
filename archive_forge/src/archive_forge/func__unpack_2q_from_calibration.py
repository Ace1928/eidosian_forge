from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING
from cirq import ops
from cirq.devices import noise_utils
from cirq_google import engine
from cirq_google import ops as cg_ops
from cirq_google.devices import google_noise_properties
from cirq_google.engine import util
def _unpack_2q_from_calibration(metric_name: str, calibration: engine.Calibration) -> Dict[Tuple['cirq.Qid', ...], float]:
    """Converts a two-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {engine.Calibration.key_to_qubits(key): engine.Calibration.value_to_float(val) for key, val in calibration[metric_name].items()}