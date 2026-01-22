import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
class Parameters_deMon(Parameters):
    """Parameters class for the calculator.
    Documented in Base_deMon.__init__

    The options here are the most important ones that the user needs to be
    aware of. Further options accepted by deMon can be set in the dictionary
    input_arguments.

    """

    def __init__(self, label='rundir', atoms=None, command=None, restart=None, basis_path=None, ignore_bad_restart_file=FileIOCalculator._deprecated, deMon_restart_path='.', title='deMon input file', scftype='RKS', forces=False, dipole=False, xc='VWN', guess='TB', print_out='MOE', basis={}, ecps={}, mcps={}, auxis={}, augment={}, input_arguments=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)