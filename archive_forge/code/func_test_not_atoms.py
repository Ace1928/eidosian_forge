import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
@pytest.mark.parametrize('bad_atoms', [None, 'a_string', [molecule('H2', vacuum=5)]])
def test_not_atoms(bad_atoms):
    """Check that passing in objects which are not
    actually Atoms objects raises a setup error """
    with pytest.raises(CalculatorSetupError):
        check_atoms_type(bad_atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(bad_atoms)
    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.calculate(atoms=bad_atoms)