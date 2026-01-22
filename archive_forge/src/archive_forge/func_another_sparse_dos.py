from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.fixture
def another_sparse_dos(self):
    return RawDOSData([8.0, 2.0, 2.0, 5.0], [1.0, 1.0, 1.0, 1.0], info={'symbol': 'H', 'number': '2'})