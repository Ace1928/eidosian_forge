import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
class ImageChunk:
    """Base Class for a file chunk which contains enough information to
    reconstruct an atoms object."""

    def build(self, **kwargs):
        """Construct the atoms object from the stored information,
        and return it"""
        pass