import pytest
import numpy as np
from ase import Atoms
def test_set_masses():
    atoms = Atoms('AgAu')
    m0 = atoms.get_masses()
    atoms.set_masses([1, None])
    assert atoms.get_masses() == pytest.approx([1, m0[1]])