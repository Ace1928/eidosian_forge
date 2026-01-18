import pytest
from ase.build import bulk
@pytest.mark.calculator_lite
def test_smearing(espresso_factory):
    atoms = bulk('Cu')
    input_data = {'system': {'occupations': 'smearing', 'smearing': 'fermi-dirac', 'degauss': 0.02}}
    atoms.calc = espresso_factory.calc(input_data=input_data)
    atoms.get_potential_energy()
    verify(atoms.calc)