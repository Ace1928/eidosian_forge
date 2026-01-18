import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
def test_props_set_consistent(forceprop):
    forceprop._setvalue('stresses', np.zeros((natoms, 6)))
    assert forceprop['stresses'].shape == (natoms, 6)