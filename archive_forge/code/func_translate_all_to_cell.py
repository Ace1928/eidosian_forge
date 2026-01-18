from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def translate_all_to_cell(self, cell=[0, 0, 0]):
    """Translate all Wannier functions to specified cell.

        Move all Wannier orbitals to a specific unit cell.  There
        exists an arbitrariness in the positions of the Wannier
        orbitals relative to the unit cell. This method can move all
        orbitals to the unit cell specified by ``cell``.  For a
        `\\Gamma`-point calculation, this has no effect. For a
        **k**-point calculation the periodicity of the orbitals are
        given by the large unit cell defined by repeating the original
        unitcell by the number of **k**-points in each direction.  In
        this case it is useful to move the orbitals away from the
        boundaries of the large cell before plotting them. For a bulk
        calculation with, say 10x10x10 **k** points, one could move
        the orbitals to the cell [2,2,2].  In this way the pbc
        boundary conditions will not be noticed.
        """
    scaled_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T * self.kptgrid / (2 * pi)
    trans_wc = np.array(cell)[None] - np.floor(scaled_wc)
    for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
        U_ww *= np.exp(2j * pi * np.dot(trans_wc, kpt_c))
    self.update()