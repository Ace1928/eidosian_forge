import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def _write_atomic_coordinates_zmatrix(self, fd, atoms):
    """Write atomic coordinates in Z-matrix format.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
    species, species_numbers = self.species(atoms)
    fd.write('\n')
    fd.write('ZM.UnitsLength   Ang\n')
    fd.write('%block Zmatrix\n')
    fd.write('  cartesian\n')
    fstr = '{:5d}' + '{:20.10f}' * 3 + '{:3d}' * 3 + '{:7d} {:s}\n'
    a2constr = self.make_xyz_constraints(atoms)
    a2p, a2s = (atoms.get_positions(), atoms.get_chemical_symbols())
    for ia, (sp, xyz, ccc, sym) in enumerate(zip(species_numbers, a2p, a2constr, a2s)):
        fd.write(fstr.format(sp, xyz[0], xyz[1], xyz[2], ccc[0], ccc[1], ccc[2], ia + 1, sym))
    fd.write('%endblock Zmatrix\n')
    origin = tuple(-atoms.get_celldisp().flatten())
    if any(origin):
        fd.write('%block AtomicCoordinatesOrigin\n')
        fd.write('     %.4f  %.4f  %.4f\n' % origin)
        fd.write('%endblock AtomicCoordinatesOrigin\n')
        fd.write('\n')