import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_rotation_from_projection(rng):
    proj_nw = rng.rand(6, 4)
    assert orthonormality_error(proj_nw[:int(min(proj_nw.shape))]) > 1
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=True)
    assert orthonormality_error(U_ww) < 1e-10, 'U_ww not unitary'
    assert orthogonality_error(C_ul.T) < 1e-10, 'C_ul columns not orthogonal'
    assert normalization_error(C_ul) < 1e-10, 'C_ul not normalized'
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=False)
    assert normalization_error(U_ww) < 1e-10, 'U_ww not normalized'