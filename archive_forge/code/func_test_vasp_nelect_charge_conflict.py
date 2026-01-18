import pytest
from ase.build import bulk
@calc('vasp')
def test_vasp_nelect_charge_conflict(factory, system, expected_nelect_from_vasp):
    charge = -2
    calc = factory.calc(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False, nelect=expected_nelect_from_vasp - charge + 1, charge=charge)
    system.calc = calc
    with pytest.raises(ValueError):
        system.get_potential_energy()