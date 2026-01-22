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
def _cmp_res(r0: Residue, r1: Residue, verbose: bool, cmpdict: Dict, rtol: float=None, atol: float=None) -> None:
    r0id, r0fid, r1fid = (r0.id, r0.full_id, r1.full_id)
    chn = r0.parent.id
    if chn not in cmpdict['chains']:
        cmpdict['chains'].append(chn)
    cmpdict['rCount'] += 1
    if r0fid == r1fid:
        cmpdict['rMatchCount'] += 1
    elif verbose:
        print(r0fid, '!=', r1fid)
    if hasattr(r0, 'internal_coord') and r0.internal_coord is not None:
        ric0 = r0.internal_coord
        ric1 = r1.internal_coord
        r0prev = sorted((ric.rbase for ric in ric0.rprev))
        r1prev = sorted((ric.rbase for ric in ric1.rprev))
        r0next = sorted((ric.rbase for ric in ric0.rnext))
        r1next = sorted((ric.rbase for ric in ric1.rnext))
        if r0prev != r1prev:
            if verbose:
                print(r0, 'rprev error:', r0prev, '!=', r1prev)
            cmpdict['rpnMismatchCount'] += 1
        if r0next != r1next:
            if verbose:
                print(r0, 'rnext error', r0next, '!=', r1next)
            cmpdict['rpnMismatchCount'] += 1
    if ' ' == r0id[0] and (not (' ' == r0.resname[0] or 2 == len(r0.resname))):
        cmpdict['residues'] += 1
        longer = r0 if len(r0.child_dict) >= len(r1.child_dict) else r1
        for ak in longer.child_dict:
            a0 = r0.child_dict.get(ak, None)
            if a0 is None:
                aknd = re.sub('D', 'H', ak, count=1)
                a0 = r0.child_dict.get(aknd, None)
            a1 = r1.child_dict.get(ak, None)
            if a1 is None:
                aknd = re.sub('D', 'H', ak, count=1)
                a1 = r1.child_dict.get(aknd, None)
            if a0 is None or a1 is None or 0 == a0.is_disordered() == a1.is_disordered():
                _cmp_atm(r0, r1, a0, a1, verbose, cmpdict, rtol=rtol, atol=atol)
            elif 2 == a0.is_disordered() == a1.is_disordered():
                cmpdict['disAtmCount'] += 1
                for da0k in a0.child_dict:
                    _cmp_atm(r0, r1, a0.child_dict.get(da0k, None), a1.child_dict.get(da0k, None), verbose, cmpdict, rtol=rtol, atol=atol)
            else:
                if verbose:
                    print('disorder disagreement:', r0.get_full_id(), ak)
                cmpdict['aCount'] += 1