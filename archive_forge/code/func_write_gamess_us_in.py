import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def write_gamess_us_in(fd, atoms, properties=None, **params):
    params = deepcopy(params)
    if properties is None:
        properties = ['energy']
    contrl = params.pop('contrl', dict())
    if 'runtyp' not in contrl:
        if 'forces' in properties:
            contrl['runtyp'] = 'gradient'
        else:
            contrl['runtyp'] = 'energy'
    xc = params.pop('xc', None)
    if xc is not None and 'dfttyp' not in contrl:
        contrl['dfttyp'] = _xc.get(xc.upper(), xc.upper())
    magmom_tot = int(round(atoms.get_initial_magnetic_moments().sum()))
    if 'mult' not in contrl:
        contrl['mult'] = abs(magmom_tot) + 1
    if 'scftyp' not in contrl:
        contrl['scftyp'] = 'rhf' if contrl['mult'] == 1 else 'uhf'
    ecp = params.pop('ecp', None)
    if ecp is not None and 'pp' not in contrl:
        contrl['pp'] = 'READ'
    basis_spec = None
    if 'basis' not in params:
        params['basis'] = dict(gbasis='N21', ngauss=3)
    else:
        keys = set(params['basis'])
        if keys.intersection(set(atoms.symbols)) or any(map(lambda x: isinstance(x, int), keys)):
            basis_spec = params.pop('basis')
    out = [_write_block('contrl', contrl)]
    out += [_write_block(*item) for item in params.items()]
    out.append(_write_geom(atoms, basis_spec))
    if ecp is not None:
        out.append(_write_ecp(atoms, ecp))
    fd.write('\n\n'.join(out))