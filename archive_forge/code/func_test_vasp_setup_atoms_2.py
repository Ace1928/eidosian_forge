import pytest
from ase.atoms import Atoms
@calc('vasp')
@pytest.mark.parametrize('settings, expected', [(dict(xc='pbe', setups={'base': 'gw'}), ('Ca_sv_GW', 'In_d_GW', 'I_GW')), (dict(xc='pbe', setups={'base': 'gw', 'I': ''}), ('Ca_sv_GW', 'In_d_GW', 'I')), (dict(xc='pbe', setups={'base': 'gw', 'Ca': '_sv', 2: 'I'}), ('Ca_sv', 'In_d_GW', 'I'))])
def test_vasp_setup_atoms_2(factory, do_check, atoms_2, settings, expected):
    do_check(factory, atoms_2, expected, settings)