import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def with_repetition_ids(self, repetition_ids: List[str]) -> 'cirq.CircuitOperation':
    """Returns a copy of this `CircuitOperation` with the given repetition IDs.

        Args:
            repetition_ids: List of new repetition IDs to use. Must have length equal to the
                existing number of repetitions.

        Returns:
            A copy of this object with `repetition_ids=repetition_ids`.
        """
    return self.replace(repetition_ids=repetition_ids)