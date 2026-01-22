from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _create_basis_state(self, index):
    """Return a computational basis state over all wires.
            Args:
                index (int): integer representing the computational basis state.
            """
    self._qubit_state.setBasisState(index)