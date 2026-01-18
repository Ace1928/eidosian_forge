import os
from subprocess import Popen, PIPE
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
def write_inp(self, atoms):
    """Write the *inp* input file of FLEUR.

        First, the information from Atoms is written to the simple input
        file and the actual input file *inp* is then generated with the
        FLEUR input generator. The location of input generator is specified
        in the environment variable FLEUR_INPGEN.

        Finally, the *inp* file is modified according to the arguments of
        the FLEUR calculator object.
        """
    with open('inp_simple', 'w') as fh:
        self._write_inp(atoms, fh)