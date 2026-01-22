import numpy as np
import warnings
from Bio.File import as_handle
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def _update_header_entry(self, target_key, keys):
    md = self._mmcif_dict
    for key in keys:
        val = md.get(key)
        try:
            item = val[0]
        except (TypeError, IndexError):
            continue
        if item != '?' and item != '.':
            self.header[target_key] = item
            break