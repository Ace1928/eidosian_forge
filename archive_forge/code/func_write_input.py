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
def write_input(self, atoms, properties=None, system_changes=None):
    """Write input (fdf)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """
    FileIOCalculator.write_input(self, atoms=atoms, properties=properties, system_changes=system_changes)
    if system_changes is None and properties is None:
        return
    filename = self.getpath(ext='fdf')
    if system_changes is not None:
        self.remove_analysis()
    with open(filename, 'w') as fd:
        fd.write(format_fdf('SystemName', self.prefix))
        fd.write(format_fdf('SystemLabel', self.prefix))
        fd.write('\n')
        fdf_arguments = self['fdf_arguments']
        keys = sorted(fdf_arguments.keys())
        for key in keys:
            fd.write(format_fdf(key, fdf_arguments[key]))
        if 'SCFMustConverge' not in fdf_arguments.keys():
            fd.write(format_fdf('SCFMustConverge', True))
        fd.write('\n')
        fd.write(format_fdf('Spin     ', self['spin']))
        if self['spin'] == 'collinear':
            fd.write(format_fdf('SpinPolarized', (True, '# Backwards compatibility.')))
        elif self['spin'] == 'non-collinear':
            fd.write(format_fdf('NonCollinear', (True, '# Backwards compatibility.')))
        functional, authors = self['xc']
        fd.write(format_fdf('XC.functional', functional))
        fd.write(format_fdf('XC.authors', authors))
        fd.write('\n')
        fd.write(format_fdf('MeshCutoff', (self['mesh_cutoff'], 'eV')))
        fd.write(format_fdf('PAO.EnergyShift', (self['energy_shift'], 'eV')))
        fd.write('\n')
        self._write_species(fd, atoms)
        self._write_structure(fd, atoms)
        if system_changes is None or ('numbers' not in system_changes and 'initial_magmoms' not in system_changes and ('initial_charges' not in system_changes)):
            fd.write(format_fdf('DM.UseSaveDM', True))
        if 'density' in properties:
            fd.write(format_fdf('SaveRho', True))
        self._write_kpts(fd)
        if self['bandpath'] is not None:
            lines = bandpath2bandpoints(self['bandpath'])
            fd.write(lines)
            fd.write('\n')