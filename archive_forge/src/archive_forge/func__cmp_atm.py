import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
def _cmp_atm(r0: Residue, r1: Residue, a0: Atom, a1: Atom, verbose: bool, cmpdict: Dict, rtol: float=None, atol: float=None) -> None:
    cmpdict['aCount'] += 1
    if a0 is None:
        if verbose:
            print(r1.get_full_id(), 'None !=', a1.get_full_id(), a1.parent.resname)
    elif a1 is None:
        if verbose:
            print(r0.get_full_id(), a0.get_full_id(), a0.parent.resname, '!= None')
    else:
        if a0.get_full_id() == a1.get_full_id() or _atmfid_d2h(a0) == a1.get_full_id():
            cmpdict['aFullIdMatchCount'] += 1
        elif verbose:
            print(r0.get_full_id(), a0.get_full_id(), a0.parent.resname, '!=', a1.get_full_id())
        ac_rslt = False
        if rtol is None and atol is None:
            a0c = np.round(a0.get_coord(), 3)
            a1c = np.round(a1.get_coord(), 3)
            ac_rslt = np.array_equal(a0c, a1c)
        else:
            a0c = a0.get_coord()
            a1c = a1.get_coord()
            ac_rslt = np.allclose(a0c, a1c, rtol=rtol, atol=atol)
        if ac_rslt:
            cmpdict['aCoordMatchCount'] += 1
        elif verbose:
            print('atom coords disagree:', r0.get_full_id(), a0.get_full_id(), a1.get_full_id(), a0c, '!=', a1c)