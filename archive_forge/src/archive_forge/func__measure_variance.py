from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
def _measure_variance(self, meas, tableau_simulator, **_):
    """Measure the variance with respect to the state of simulator device."""
    meas_obs = qml.operation.convert_to_opmath(meas.obs)
    meas_obs1 = meas_obs.simplify()
    meas_obs2 = (meas_obs1 ** 2).simplify()
    return self._measure_expectation(ExpectationMP(meas_obs2), tableau_simulator) - self._measure_expectation(ExpectationMP(meas_obs1), tableau_simulator) ** 2