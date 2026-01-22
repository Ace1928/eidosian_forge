from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
class MinimalDOSData(DOSData):
    """Inherit from ABC to test its features"""

    def get_energies(self):
        return NotImplementedError()

    def get_weights(self):
        return NotImplementedError()

    def copy(self):
        return NotImplementedError()