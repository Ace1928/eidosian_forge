import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
def _assign_atom_mass(self):
    """Return atom weight (PRIVATE)."""
    try:
        return IUPACData.atom_weights[self.element.capitalize()]
    except (AttributeError, KeyError):
        return float('NaN')