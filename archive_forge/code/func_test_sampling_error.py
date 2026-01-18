from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
def test_sampling_error(self, sparse_dos):
    with pytest.raises(ValueError):
        sparse_dos._sample([1, 2, 3], width=0.0)
    with pytest.raises(ValueError):
        sparse_dos._sample([1, 2, 3], width=-1)