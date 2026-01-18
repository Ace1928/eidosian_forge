import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
Writes NWChem input file.

    Parameters
    ----------
    fd
        file descriptor
    atoms
        atomic configuration
    properties
        list of properties to compute; by default only the
        calculation of the energy is requested
    echo
        if True include the `echo` keyword at the top of the file,
        which causes the content of the input file to be included
        in the output file
    params
        dict of instructions blocks to be included
    