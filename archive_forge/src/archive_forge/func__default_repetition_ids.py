import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def _default_repetition_ids(self) -> Optional[List[str]]:
    return default_repetition_ids(self.repetitions) if self.use_repetition_ids else None