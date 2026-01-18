from io import StringIO
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import InputError, ReadError
from ase.calculators.calculator import CalculatorSetupError
import multiprocessing
from ase import io
import numpy as np
import json
from ase.units import Bohr, Hartree
import warnings
import os
Read psi4 outputs made from this ASE calculator
        