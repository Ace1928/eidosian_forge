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
def _write_atomic_coordinates_xyz(self, fd, atoms):
    """Write atomic coordinates.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
    species, species_numbers = self.species(atoms)
    fd.write('\n')
    fd.write('AtomicCoordinatesFormat  Ang\n')
    fd.write('%block AtomicCoordinatesAndAtomicSpecies\n')
    for atom, number in zip(atoms, species_numbers):
        xyz = atom.position
        line = ('    %.9f' % xyz[0]).rjust(16) + ' '
        line += ('    %.9f' % xyz[1]).rjust(16) + ' '
        line += ('    %.9f' % xyz[2]).rjust(16) + ' '
        line += str(number) + '\n'
        fd.write(line)
    fd.write('%endblock AtomicCoordinatesAndAtomicSpecies\n')
    fd.write('\n')
    origin = tuple(-atoms.get_celldisp().flatten())
    if any(origin):
        fd.write('%block AtomicCoordinatesOrigin\n')
        fd.write('     %.4f  %.4f  %.4f\n' % origin)
        fd.write('%endblock AtomicCoordinatesOrigin\n')
        fd.write('\n')