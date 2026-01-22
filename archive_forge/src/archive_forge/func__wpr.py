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
def _wpr(entity, fp, pdbid, chainid, picFlags: int=IC_Residue.picFlagsDefault, hCut: Optional[Union[float, None]]=None, pCut: Optional[Union[float, None]]=None):
    if entity.internal_coord:
        if not chainid or not pdbid:
            chain = entity.parent
            if not chainid:
                chainid = chain.id
            if not pdbid:
                struct = chain.parent.parent
                pdbid = struct.header.get('idcode')
        fp.write(entity.internal_coord._write_PIC(pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut))
    else:
        fp.write(IC_Residue._residue_string(entity))