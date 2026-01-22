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
def _create_edra(self, verbose: bool=False) -> None:
    """Create IC_Chain and IC_Residue di/hedra for atom coordinates.

        AllBonds handled here.

        :param bool verbose: default False.
            Warn about missing N, Ca, C backbone atoms.
        """
    if not self.ak_set:
        return
    sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
    if self.lc != 'G':
        sCB = self.rak('CB')
    if 0 < len(self.rnext) and self.rnext[0].ak_set:
        for rn in self.rnext:
            nN, nCA, nC = (rn.rak('N'), rn.rak('CA'), rn.rak('C'))
            nextNCaC = rn.split_akl((nN, nCA, nC), missingOK=True)
            for tpl in nextNCaC:
                for ak in tpl:
                    if ak in rn.ak_set:
                        self.ak_set.add(ak)
                    else:
                        for rn_ak in rn.ak_set:
                            if rn_ak.altloc_match(ak):
                                self.ak_set.add(rn_ak)
            self._gen_edra((sN, sCA, sC, nN))
            self._gen_edra((sCA, sC, nN, nCA))
            self._gen_edra((sC, nN, nCA, nC))
            self._gen_edra((sCA, sC, nN))
            self._gen_edra((sC, nN, nCA))
            self._gen_edra((nN, nCA, nC))
            try:
                nO = rn.akc['O']
            except KeyError:
                nCB = rn.akc.get('CB', None)
                if nCB is not None and nCB in rn.ak_set:
                    self.ak_set.add(nCB)
                    self._gen_edra((nN, nCA, nCB))
                    self._gen_edra((sC, nN, nCA, nCB))
    if 0 == len(self.rprev):
        self._gen_edra((sN, sCA, sC))
    backbone = ic_data_backbone
    for edra in backbone:
        if all((atm in self.akc for atm in edra)):
            r_edra = [self.rak(atom) for atom in edra]
            self._gen_edra(r_edra)
    if self.lc is not None:
        sidechain = ic_data_sidechains.get(self.lc, [])
        for edraLong in sidechain:
            edra = edraLong[0:4]
            if all((atm in self.akc for atm in edra)):
                r_edra = [self.rak(atom) for atom in edra]
                self._gen_edra(r_edra)
        if IC_Residue._AllBonds:
            sidechain = ic_data_sidechain_extras.get(self.lc, [])
            for edra in sidechain:
                if all((atm in self.akc for atm in edra)):
                    r_edra = [self.rak(atom) for atom in edra]
                    self._gen_edra(r_edra)
    if self.gly_Cbeta and 'G' == self.lc:
        self.ak_set.add(AtomKey(self, 'CB'))
        sCB = self.rak('CB')
        sCB.missing = False
        self.cic.akset.add(sCB)
        sO = self.rak('O')
        htpl = (sCB, sCA, sC)
        self._gen_edra(htpl)
        dtpl = (sO, sC, sCA, sCB)
        self._gen_edra(dtpl)
        d = self.dihedra[dtpl]
        d.ric = self
        d._set_hedra()
        if not hasattr(self.cic, 'gcb'):
            self.cic.gcb = {}
        self.cic.gcb[sCB] = dtpl
    self._link_dihedra(verbose)
    if verbose:
        self.rak('O')
        missing = []
        for akk, akv in self.akc.items():
            if isinstance(akk, str) and akv.missing:
                missing.append(akv)
        if missing:
            chn = self.residue.parent
            chn_id = chn.id
            chn_len = len(chn.internal_coord.ordered_aa_ic_list)
            print(f'chain {chn_id} len {chn_len} missing atom(s): {missing}')