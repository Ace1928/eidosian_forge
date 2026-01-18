import pytest
from ase.atoms import Atoms
@calc('vasp')
@pytest.mark.parametrize('settings, expected', [(dict(xc='pbe'), ('Ca_sv', 'Gd', 'Cs_sv')), (dict(xc='pbe', setups='recommended'), ('Ca_sv', 'Gd_31', 'Cs_sv')), (dict(xc='pbe', setups='materialsproject'), ('Ca_sv', 'Gd', 'Cs'))])
def test_setup_error(factory, do_check, atoms_1, settings, expected):
    """Do a test, where we purposely make mistakes"""
    do_check(factory, atoms_1, expected, settings, should_raise=True)