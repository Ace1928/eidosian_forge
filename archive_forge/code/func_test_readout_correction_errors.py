import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_readout_correction_errors():
    kwargs = _get_ZZ_Z_Z_bsa_constructor_args()
    settings = kwargs['simul_settings']
    ro_bsa, _, _ = _get_mock_readout_calibration()
    kwargs['readout_calibration'] = ro_bsa
    bsa = cw.BitstringAccumulator(**kwargs)
    np.testing.assert_allclose(bsa.means(), [0, 0, 0])
    assert bsa.variance(settings[0]) == np.inf