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
class PhasedFSimCalibrationOptions(abc.ABC, Generic[RequestT]):
    """Base class for calibration-specific options passed together with the requests."""

    @abc.abstractmethod
    def create_phased_fsim_request(self, pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...], gate: cirq.Gate) -> RequestT:
        """Create a PhasedFSimCalibrationRequest of the correct type for these options.

        Args:
            pairs: Set of qubit pairs to characterize. A single qubit can appear on at most one
                pair in the set.
            gate: Gate to characterize for each qubit pair from pairs. This must be a supported gate
                which can be described cirq.PhasedFSim gate. This gate must be serialized by the
                cirq_google.SerializableGateSet used
        """