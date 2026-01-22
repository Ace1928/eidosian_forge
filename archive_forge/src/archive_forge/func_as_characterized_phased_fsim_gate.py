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
def as_characterized_phased_fsim_gate(self, parameters: PhasedFSimCharacterization) -> cirq.PhasedFSimGate:
    """Creates a PhasedFSimGate which represents the characterized engine_gate but includes
        deviations in unitary parameters.

        Args:
            parameters: The results of characterization of the engine gate.

        Returns:
            Instance of PhasedFSimGate that executes a gate according to the characterized
            parameters of the engine_gate.
        """
    assert parameters.chi is not None
    return cirq.PhasedFSimGate(theta=parameters.theta or 0.0, zeta=parameters.zeta or 0.0, chi=parameters.chi - 2 * np.pi * self.phase_exponent, gamma=parameters.gamma or 0.0, phi=parameters.phi or 0.0)