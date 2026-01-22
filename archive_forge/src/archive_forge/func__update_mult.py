import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _update_mult(magmom_tot, **params):
    theory = params['theory']
    if magmom_tot == 0:
        magmom_mult = 1
    else:
        magmom_mult = np.sign(magmom_tot) * (abs(magmom_tot) + 1)
    if 'scf' in params:
        for kw in ['nopen', 'singlet', 'doublet', 'triplet', 'quartet', 'quintet', 'sextet', 'septet', 'octet']:
            if kw in params['scf']:
                break
        else:
            params['scf']['nopen'] = magmom_tot
    elif theory in ['scf', 'mp2', 'ccsd', 'tce']:
        params['scf'] = dict(nopen=magmom_tot)
    if 'dft' in params:
        if 'mult' not in params['dft']:
            params['dft']['mult'] = magmom_mult
    elif theory in ['dft', 'tddft']:
        params['dft'] = dict(mult=magmom_mult)
    if 'nwpw' in params:
        if 'mult' not in params['nwpw']:
            params['nwpw']['mult'] = magmom_mult
    elif theory in ['pspw', 'band', 'paw']:
        params['nwpw'] = dict(mult=magmom_mult)
    return params