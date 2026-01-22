import re
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio import BiopythonWarning
def accept_atom(self, atom):
    """Verify if atoms are not Hydrogen."""
    name = atom.get_id()
    if _hydrogen.match(name):
        return 0
    else:
        return 1