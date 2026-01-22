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
def any_none(self) -> bool:
    """Returns True if any the angle is None"""
    return self.theta is None or self.zeta is None or self.chi is None or (self.gamma is None) or (self.phi is None)