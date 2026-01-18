import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert

    Manually read a lammpsdata file and grep for the different
    quantities we want to check.  Accepts either a string indicating the name
    of the file, a pathlib.Path object indicating the location of the file, a
    StringIO object containing the file contents, or a file object
    