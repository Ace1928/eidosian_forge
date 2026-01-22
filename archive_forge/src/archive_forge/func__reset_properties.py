import numpy as np
from Bio.PDB.PDBExceptions import PDBException
def _reset_properties(self):
    """Reset all relevant properties to None to avoid conflicts between runs."""
    self.reference_coords = None
    self.coords = None
    self.transformed_coords = None
    self.rot = None
    self.tran = None
    self.rms = None
    self.init_rms = None