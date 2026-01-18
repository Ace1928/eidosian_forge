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
def to_calibration_layer(self) -> CalibrationLayer:
    circuit = cirq.Circuit((self.gate.on(*pair) for pair in self.pairs))
    return CalibrationLayer(calibration_type=_XEB_PHASED_FSIM_HANDLER_NAME, program=circuit, args=self.options.to_args())