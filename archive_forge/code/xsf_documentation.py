import numpy as np
from ase.atoms import Atoms
from ase.units import Hartree
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import writer, reader
Yield images and optionally data from xsf file.

    Yields image1, image2, ..., imageN[, data].

    Images are Atoms objects and data is a numpy array.

    Presently supports only a single 3D datagrid.