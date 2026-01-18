import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_vasp_no_cell(testdir):
    """Check missing cell handling."""
    atoms = molecule('CH4')
    assert atoms.cell.rank == 0
    with pytest.raises(CalculatorSetupError):
        check_cell(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)
    with pytest.raises(RuntimeError):
        atoms.write('POSCAR')
    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_total_energy()