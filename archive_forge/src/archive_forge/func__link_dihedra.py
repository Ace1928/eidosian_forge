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
def _link_dihedra(self, verbose: bool=False) -> None:
    """Housekeeping after loading all residues and dihedra.

        - Link dihedra to this residue
        - form id3_dh_index
        - form ak_set
        - set NCaCKey to be available AtomKeys

        called for loading PDB / atom coords
        """
    for dh in self.dihedra.values():
        dh.ric = self
        dh.cic = self.cic
        self.ak_set.update(dh.atomkeys)
    for h in self.hedra.values():
        self.ak_set.update(h.atomkeys)
        h.cic = self.cic
    if not self.akc:
        self._build_rak_cache()
    self.NCaCKey = []
    self.NCaCKey.extend(self.split_akl((AtomKey(self, 'N'), AtomKey(self, 'CA'), AtomKey(self, 'C'))))