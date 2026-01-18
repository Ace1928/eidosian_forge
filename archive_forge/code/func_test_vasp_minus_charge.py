import pytest
from ase.build import bulk
@calc('vasp')
def test_vasp_minus_charge(factory, system, expected_nelect_from_vasp):
    charge = -2
    calc = factory.calc(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False, charge=charge)
    calc.initialize(system)
    calc.write_input(system)
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] == expected_nelect_from_vasp - charge