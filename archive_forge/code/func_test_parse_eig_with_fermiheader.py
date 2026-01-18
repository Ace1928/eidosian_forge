from io import StringIO
import numpy as np
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.abinit import read_abinit_out, read_eig, match_kpt_header
from ase.units import Hartree, Bohr
def test_parse_eig_with_fermiheader():
    eigval_ref = np.array([[-0.2, 0.2, 0.3], [-0.3, 0.4, 0.5]]).reshape(1, 2, 3)
    kpts_ref = np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    weights_ref = [0.1, 0.2]
    eig_buf = StringIO(eig_text)
    data = read_eig(eig_buf)
    assert data['eigenvalues'] / Hartree == pytest.approx(eigval_ref)
    assert data['ibz_kpoints'] == pytest.approx(kpts_ref)
    assert data['kpoint_weights'] == pytest.approx(weights_ref)
    assert data['fermilevel'] / Hartree == pytest.approx(0.123)