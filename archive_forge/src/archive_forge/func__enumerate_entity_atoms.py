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
def _enumerate_entity_atoms(entity):
    need = False
    for atm in entity.get_atoms():
        need = not atm.get_serial_number()
        break
    if need:
        anum = 1
        for res in entity.get_residues():
            if 2 == res.is_disordered():
                for r in res.child_dict.values():
                    for atm in r.get_unpacked_list():
                        atm.set_serial_number(anum)
                        anum += 1
            else:
                for atm in res.get_unpacked_list():
                    atm.set_serial_number(anum)
                    anum += 1