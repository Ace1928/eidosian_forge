import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_vasp_get_calculator():
    cls_ = get_calculator_class('vasp')
    assert cls_ == Vasp
    assert get_calculator_class(Vasp.name) == cls_