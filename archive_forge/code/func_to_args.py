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
def to_args(self) -> Dict[str, Any]:
    """Convert this dataclass to an `args` dictionary suitable for sending to the Quantum
        Engine calibration API."""
    args: Dict[str, Any] = {'n_library_circuits': self.n_library_circuits, 'n_combinations': self.n_combinations, 'cycle_depths': '_'.join((f'{cd:d}' for cd in self.cycle_depths))}
    if self.fatol is not None:
        args['fatol'] = self.fatol
    if self.xatol is not None:
        args['xatol'] = self.xatol
    fsim_options = dataclasses.asdict(self.fsim_options)
    fsim_options = {k: v for k, v in fsim_options.items() if v is not None}
    args.update(fsim_options)
    return args