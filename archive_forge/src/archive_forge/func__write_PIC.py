import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def _write_PIC(self, pdbid: str='0PDB', chainid: str='A', picFlags: int=picFlagsDefault, hCut: Optional[Union[float, None]]=None, pCut: Optional[Union[float, None]]=None) -> str:
    """Write PIC format lines for this residue.

        See :func:`.PICIO.write_PIC`.

        :param str pdbid: PDB idcode string; default 0PDB
        :param str chainid: PDB Chain ID character; default A
        :param int picFlags: control details written to PIC file; see
            :meth:`.PICIO.write_PIC`
        :param float hCut: only write hedra with ref db angle std dev > this
            value; default None
        :param float pCut: only write primary dihedra with ref db angle
            std dev > this value; default None
        """
    pAcc = IC_Residue.pic_accuracy
    if pdbid is None:
        pdbid = '0PDB'
    if chainid is None:
        chainid = 'A'
    icr = IC_Residue
    s = icr._residue_string(self.residue)
    if picFlags & icr.pic_flags.initAtoms and 0 == len(self.rprev) and hasattr(self, 'NCaCKey') and (self.NCaCKey is not None) and (not np.all(self.residue['N'].coord == self.residue['N'].coord[0])):
        NCaChedron = self.pick_angle(self.NCaCKey[0])
        if NCaChedron is not None:
            try:
                ts = IC_Residue._pdb_atom_string(self.residue['N'], cif_extend=True)
                ts += IC_Residue._pdb_atom_string(self.residue['CA'], cif_extend=True)
                ts += IC_Residue._pdb_atom_string(self.residue['C'], cif_extend=True)
                s += ts
            except KeyError:
                pass
    base = pdbid + ' ' + chainid + ' '
    cic = self.cic
    if picFlags & icr.pic_flags.hedra or picFlags & icr.pic_flags.tau:
        for h in sorted(self.hedra.values()):
            if not picFlags & icr.pic_flags.hedra and picFlags & icr.pic_flags.tau and (h.e_class != 'NCAC'):
                continue
            if hCut is not None:
                hc = h.xrh_class if hasattr(h, 'xrh_class') else h.e_class
                if hc in hedra_defaults and hedra_defaults[hc][1] <= hCut:
                    continue
            hndx = h.ndx
            try:
                s += base + h.id + ' ' + f'{cic.hedraL12[hndx]:{pAcc}} {cic.hedraAngle[hndx]:{pAcc}} {cic.hedraL23[hndx]:{pAcc}}' + '\n'
            except KeyError:
                pass
    for d in sorted(self.dihedra.values()):
        if d.primary:
            if not picFlags & icr.pic_flags.primary:
                if not picFlags & d.bits():
                    continue
        elif not picFlags & icr.pic_flags.secondary:
            continue
        if pCut is not None:
            if d.primary and d.pclass in dihedra_primary_defaults and (dihedra_primary_defaults[d.pclass][1] <= pCut):
                continue
        try:
            s += base + d.id + ' ' + f'{cic.dihedraAngle[d.ndx]:{pAcc}}' + '\n'
        except KeyError:
            pass
    if picFlags & icr.pic_flags.bFactors:
        col = 0
        for a in sorted(self.residue.get_atoms()):
            if 2 == a.is_disordered():
                if IC_Residue.no_altloc or self.alt_ids is None:
                    s, col = self._write_pic_bfac(a.selected_child, s, col)
                else:
                    for atm in a.child_dict.values():
                        s, col = self._write_pic_bfac(atm, s, col)
            else:
                s, col = self._write_pic_bfac(a, s, col)
        if 0 != col % 5:
            s += '\n'
    return s