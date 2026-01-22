import json
import logging
import warnings
from json import JSONEncoder
from typing import (
from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
from pyquil.experiment._program import (
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit

        Generate another ``Experiment`` object that can be used to calibrate the various multi-qubit
        observables involved in this ``Experiment``. This is achieved by preparing the plus-one
        (minus-one) eigenstate of each ``out_operator``, and measuring the resulting expectation
        value of the same ``out_operator``. Ideally, this would always give +1 (-1), but when
        symmetric readout error is present the effect is to scale the resultant expectations by some
        constant factor. Determining this scale factor is what we call *readout calibration*, and
        then the readout error in subsequent measurements can then be mitigated by simply dividing
        by the scale factor.

        :return: A new ``Experiment`` that can calibrate the readout error of all the
            observables involved in this experiment.
        