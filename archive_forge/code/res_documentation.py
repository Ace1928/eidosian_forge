import glob
import re
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

        Writes Res to a file. The supported kwargs are the same as those for
        the Res.get_string method and are passed through directly.
        