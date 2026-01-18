import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
Test explicitly passing a command to the calculator