import warnings
import numpy as np
from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
def _handle_PDB_exception(self, message, line_counter):
    """Handle exception (PRIVATE).

        This method catches an exception that occurs in the StructureBuilder
        object (if PERMISSIVE), or raises it again, this time adding the
        PDB line number to the error message.
        """
    message = '%s at line %i.' % (message, line_counter)
    if self.PERMISSIVE:
        warnings.warn('PDBConstructionException: %s\nException ignored.\nSome atoms or residues may be missing in the data structure.' % message, PDBConstructionWarning)
    else:
        raise PDBConstructionException(message) from None