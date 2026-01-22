import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def ake_recurse(akList: List) -> List:
    """Bulid combinatorics of AtomKey lists."""
    car = akList[0]
    if len(akList) > 1:
        retList = []
        for ak in car:
            cdr = akList[1:]
            rslt = ake_recurse(cdr)
            for r in rslt:
                r.insert(0, ak)
                retList.append(r)
        return retList
    elif len(car) == 1:
        return [list(car)]
    else:
        retList = [[ak] for ak in car]
        return retList