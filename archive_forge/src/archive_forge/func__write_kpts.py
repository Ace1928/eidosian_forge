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
def _write_kpts(self, fd):
    """Write kpts.

        Parameters:
            - f : Open filename.
        """
    if self['kpts'] is None:
        return
    kpts = np.array(self['kpts'])
    fd.write('\n')
    fd.write('#KPoint grid\n')
    fd.write('%block kgrid_Monkhorst_Pack\n')
    for i in range(3):
        s = ''
        if i < len(kpts):
            number = kpts[i]
            displace = 0.0
        else:
            number = 1
            displace = 0
        for j in range(3):
            if j == i:
                write_this = number
            else:
                write_this = 0
            s += '     %d  ' % write_this
        s += '%1.1f\n' % displace
        fd.write(s)
    fd.write('%endblock kgrid_Monkhorst_Pack\n')
    fd.write('\n')