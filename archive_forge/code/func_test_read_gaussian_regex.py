import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_read_gaussian_regex():
    """ Test regex used in read_gaussian_in"""
    link0_line = '%chk=example.chk'
    link0_match = _re_link0.match(link0_line)
    assert link0_match.group(1) == 'chk'
    assert link0_match.group(2) == 'example.chk'
    link0_line = '%chk'
    link0_match = _re_link0.match(link0_line)
    assert link0_match.group(1) == 'chk'
    assert link0_match.group(2) is None
    output_type_lines = ['#P B3LYP', ' #P', '# P']
    for line in output_type_lines:
        output_type_match = _re_output_type.match(line)
        assert output_type_match.group(1) == 'P'
    method_basis_line = 'g1/Gen/TZVPFit ! ASE formatted method and basis'
    method_basis_match = _re_method_basis.match(method_basis_line)
    assert method_basis_match.group(1) == 'g1'
    assert method_basis_match.group(2) == 'Gen'
    assert method_basis_match.group(4) == 'TZVPFit '
    assert method_basis_match.group(5) == '! ASE formatted method and basis'
    method_basis_line = 'g1/Gen ! ASE formatted method and basis'
    method_basis_match = _re_method_basis.match(method_basis_line)
    assert method_basis_match.group(1) == 'g1'
    assert method_basis_match.group(2) == 'Gen '
    assert method_basis_match.group(5) == '! ASE formatted method and basis'
    chgmult_lines = ['0 1', ' 0 1', '0, 2']
    for line in chgmult_lines:
        assert _re_chgmult.match(line).group(0) == line
    nuclear_props = '(iso=0.1134289259, NMagM=-8.89, ZEff=-1)'
    nuclear_prop_line = '1{}, -0.464,   1.137,   0.0'.format(nuclear_props)
    assert _re_nuclear_props.search(nuclear_prop_line).group(0) == nuclear_props