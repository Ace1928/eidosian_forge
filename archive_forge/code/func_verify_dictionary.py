import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
def verify_dictionary(atoms, dictionary, dictionary_name):
    """
    Verify a dictionary have a key for each symbol present in the atoms object.

    Parameters:

    dictionary: dict
        Dictionary to be checked.


    dictionary_name: dict
        Name of the dictionary to be displayed in the error message.

    cell: cell object
        cell to be checked.


    Raise a ``ValueError`` if the key doesn't match the atoms present in the
    cell.
    """
    for key in set(atoms.symbols):
        if key not in dictionary:
            raise ValueError('Missing the {} key in the `{}` dictionary.'.format(key, dictionary_name))