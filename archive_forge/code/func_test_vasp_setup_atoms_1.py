import pytest
from ase.atoms import Atoms
@calc('vasp')
@pytest.mark.parametrize('settings, expected', [(dict(xc='pbe'), ('Ca_pv', 'Gd', 'Cs_sv')), (dict(xc='pbe', setups='recommended'), ('Ca_sv', 'Gd_3', 'Cs_sv')), (dict(xc='pbe', setups='materialsproject'), ('Ca_sv', 'Gd', 'Cs_sv'))])
def test_vasp_setup_atoms_1(factory, do_check, atoms_1, settings, expected):
    """
    Run some tests to ensure that VASP calculator constructs correct POTCAR files

    """
    do_check(factory, atoms_1, expected, settings)