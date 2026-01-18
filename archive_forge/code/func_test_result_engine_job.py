import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
@mock.patch('cirq_google.engine.engine_client.EngineClient')
def test_result_engine_job(_client):
    result = PhasedFSimCalibrationResult(parameters={}, gate=cirq.FSimGate(theta=np.pi / 4, phi=0.0), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, project_id='project_id', program_id='program_id', job_id='job_id')
    assert result.engine_job.project_id == 'project_id'
    assert result.engine_job.program_id == 'program_id'
    assert result.engine_job.job_id == 'job_id'