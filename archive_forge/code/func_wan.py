import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
@pytest.fixture
def wan(rng, std_calculator):

    def _wan(gpts=(8, 8, 8), atoms=None, calc=None, nwannier=2, fixedstates=None, initialwannier='bloch', kpts=(1, 1, 1), file=None, rng=rng, full_calc=False, std_calc=True):
        if std_calc and calc is None and (atoms is None):
            calc = std_calculator
        else:
            if calc is None:
                gpaw = pytest.importorskip('gpaw')
                calc = gpaw.GPAW(gpts=gpts, nbands=nwannier, kpts=kpts, symmetry='off', txt=None)
            if atoms is None and (not full_calc):
                pbc = (np.array(kpts) > 1).any()
                atoms = molecule('H2', pbc=pbc)
                atoms.center(vacuum=3.0)
            if not full_calc:
                atoms.calc = calc
                atoms.get_potential_energy()
        return Wannier(nwannier=nwannier, fixedstates=fixedstates, calc=calc, initialwannier=initialwannier, file=None, rng=rng)
    return _wan