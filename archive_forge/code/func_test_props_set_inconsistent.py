import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
def test_props_set_inconsistent(forceprop):
    with pytest.raises(ValueError):
        forceprop._setvalue('stresses', np.zeros((natoms + 2, 6)))