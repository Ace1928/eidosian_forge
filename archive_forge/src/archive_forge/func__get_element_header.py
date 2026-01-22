import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
def _get_element_header(self, atom_type, number, atom_type_number, occupancy, RMS):
    return '{0}\n{1} {2} {3} {4:.3g}\n'.format(atom_type, number, atom_type_number, occupancy, RMS)