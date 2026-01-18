from io import StringIO
import numpy as np
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.abinit import read_abinit_out, read_eig, match_kpt_header
from ase.units import Hartree, Bohr
def test_read_abinit_output():
    fd = StringIO(sample_outfile)
    results = read_abinit_out(fd)
    assert results.pop('version') == '8.0.8'
    atoms = results.pop('atoms')
    assert all(atoms.symbols == 'OO')
    assert atoms.positions == pytest.approx(np.array([[2.5, 2.5, 3.7], [2.5, 2.5, 2.5]]))
    assert all(atoms.pbc)
    assert atoms.cell[:] == pytest.approx(np.array([[5.0, 0.0, 0.1], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]]))
    ref_stress = pytest.approx([2.3, 2.4, 2.5, 3.1, 3.2, 3.3])
    assert results.pop('stress') / (Hartree / Bohr ** 3) == ref_stress
    assert results.pop('forces') == pytest.approx(np.array([[-0.1, -0.3, 0.4], [-0.2, -0.4, -0.5]]))
    for name in ('energy', 'free_energy'):
        assert results.pop(name) / Hartree == -42.5
    assert not results