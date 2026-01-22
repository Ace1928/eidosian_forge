from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
@staticmethod
def _check_adjdiff_supported_measurements(measurements: List[MeasurementProcess]):
    """Check whether given list of measurement is supported by adjoint_differentiation.

            Args:
                measurements (List[MeasurementProcess]): a list of measurement processes to check.

            Returns:
                Expectation or State: a common return type of measurements.
            """
    if not measurements:
        return None
    if len(measurements) == 1 and measurements[0].return_type is State:
        return State
    if any((measurement.return_type is not Expectation for measurement in measurements)):
        raise QuantumFunctionError('Adjoint differentiation method does not support expectation return type mixed with other return types')
    for measurement in measurements:
        if isinstance(measurement.obs, Tensor):
            if any((isinstance(obs, Projector) for obs in measurement.obs.non_identity_obs)):
                raise QuantumFunctionError('Adjoint differentiation method does not support the Projector observable')
        elif isinstance(measurement.obs, Projector):
            raise QuantumFunctionError('Adjoint differentiation method does not support the Projector observable')
    return Expectation