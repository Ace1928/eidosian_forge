import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_translate_to_cell(wan, std_calculator):
    nwannier = 2
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    for i in range(nwannier):
        wanf.translate_to_cell(w=i, cell=[0, 0, 0])
        c0_w = wanf.get_centers()
        assert (c0_w[i] < atoms.cell.array.diagonal()).all()
        wanf.translate_to_cell(w=i, cell=[1, 1, 1])
        c1_w = wanf.get_centers()
        assert (c1_w[i] > atoms.cell.array.diagonal()).all()
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c0_w = np.delete(c0_w, i, 0)
        c1_w = np.delete(c1_w, i, 0)
        assert c0_w == pytest.approx(c1_w)