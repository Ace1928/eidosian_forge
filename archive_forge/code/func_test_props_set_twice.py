import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
def test_props_set_twice(forceprop):
    with pytest.raises(ValueError):
        forceprop._setvalue('forces', np.zeros((natoms, 3)))