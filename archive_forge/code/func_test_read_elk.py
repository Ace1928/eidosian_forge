import io
import re
import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from ase.io.elk import parse_elk_eigval, read_elk
from ase.units import Hartree, Bohr
def test_read_elk():
    atoms = read_elk(io.StringIO(elk_geometry_out))
    assert str(atoms.symbols) == 'Si2'
    assert all(atoms.pbc)
    assert atoms.cell / Bohr == pytest.approx(np.array([[1.0, 0.1, 0.2], [0.3, 2.0, 0.4], [0.5, 0.6, 3.0]]))
    assert atoms.get_scaled_positions() == pytest.approx(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))