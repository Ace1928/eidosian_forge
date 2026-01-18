import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def write_kpoints(self, atoms=None, directory='./', **kwargs):
    """Writes the KPOINTS file."""
    if atoms is None:
        atoms = self.atoms
    if self.float_params['kspacing'] is not None:
        if self.float_params['kspacing'] > 0:
            return
        else:
            raise ValueError('KSPACING value {0} is not allowable. Please use None or a positive number.'.format(self.float_params['kspacing']))
    p = self.input_params
    with open(join(directory, 'KPOINTS'), 'w') as kpoints:
        kpoints.write('KPOINTS created by Atomic Simulation Environment\n')
        if isinstance(p['kpts'], dict):
            p['kpts'] = kpts2ndarray(p['kpts'], atoms=atoms)
            p['reciprocal'] = True
        shape = np.array(p['kpts']).shape
        if shape == ():
            p['kpts'] = [p['kpts']]
            shape = (1,)
        if len(shape) == 1:
            kpoints.write('0\n')
            if shape == (1,):
                kpoints.write('Auto\n')
            elif p['gamma']:
                kpoints.write('Gamma\n')
            else:
                kpoints.write('Monkhorst-Pack\n')
            [kpoints.write('%i ' % kpt) for kpt in p['kpts']]
            kpoints.write('\n0 0 0\n')
        elif len(shape) == 2:
            kpoints.write('%i \n' % len(p['kpts']))
            if p['reciprocal']:
                kpoints.write('Reciprocal\n')
            else:
                kpoints.write('Cartesian\n')
            for n in range(len(p['kpts'])):
                [kpoints.write('%f ' % kpt) for kpt in p['kpts'][n]]
                if shape[1] == 4:
                    kpoints.write('\n')
                elif shape[1] == 3:
                    kpoints.write('1.0 \n')