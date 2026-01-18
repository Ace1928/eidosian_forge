import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_write_gaussian_calc():
    """ Tests writing an input file for a Gaussian calculator. Reads this
    back in and checks that we get the parameters we expect.

    This allows testing of 'addsec', 'extra', 'ioplist',
    which we weren't able to test by reading and then writing files."""
    atoms = Atoms('H2', [[0, 0, 0], [0, 0, 0.74]])
    params = {'mem': '1GB', 'charge': 0, 'mult': 1, 'xc': 'PBE', 'save': None, 'basis': 'EPR-III', 'scf': 'qc', 'ioplist': ['1/2', '2/3'], 'freq': 'readiso', 'addsec': '297 3 1', 'extra': 'Opt = Tight'}
    atoms.calc = Gaussian(**params)
    params_expected = {}
    for k, v in params.items():
        if v is None:
            params_expected[k] = ''
        elif type(v) in [list, int]:
            params_expected[k] = v
        else:
            params_expected[k] = v.lower()
    params_expected['output_type'] = 'p'
    params_expected.pop('xc')
    params_expected['method'] = 'pbepbe'
    params_expected.pop('extra')
    params_expected['opt'] = 'tight'
    params_expected['freq'] = None
    params_expected.pop('addsec')
    params_expected['temperature'] = '297'
    params_expected['pressure'] = '3'
    params_expected['scale'] = '1'
    ioplist = params_expected.pop('ioplist')
    ioplist_txt = ''
    for iop in ioplist:
        ioplist_txt += iop + ', '
    ioplist_txt = ioplist_txt.strip(', ')
    params_expected['iop'] = ioplist_txt
    params_expected['forces'] = None
    _test_write_gaussian(atoms, params_expected, properties='forces')
    calc = Gaussian(basis='gen')
    with pytest.raises(InputError):
        calc.write_input(atoms)
    basisfilename = 'basis.txt'
    with open(basisfilename, 'w+') as fd:
        fd.write(_basis_set_text)
    calc = Gaussian(basisfile=basisfilename, output_type='p', mult=0, charge=1, basis='gen')
    atoms.calc = calc
    params_expected = calc.parameters
    params_expected['basis_set'] = _basis_set_text
    _test_write_gaussian(atoms, params_expected, properties='forces')