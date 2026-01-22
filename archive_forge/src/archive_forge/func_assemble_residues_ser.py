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
def assemble_residues_ser(self, verbose: bool=False, start: Optional[int]=None, fin: Optional[int]=None) -> None:
    """Generate IC_Residue atom coords from internal coordinates (serial).

        See :meth:`.assemble_residues` for 'numpy parallel' version.

        Filter positions between start and fin if set, find appropriate start
        coordinates for each residue and pass to :meth:`.assemble`

        :param bool verbose: default False.
            Describe runtime problems
        :param int start,fin: default None.
            Sequence position for begin, end of subregion to generate coords
            for.
        """
    self.dcsValid[:] = False
    for ric in self.ordered_aa_ic_list:
        if fin and fin < ric.residue.id[1] or (start and start > ric.residue.id[1]):
            ric.ak_set = None
            ric.akc = None
            ric.residue.child_dict = {}
            ric.residue.child_list = []
            continue
        atom_coords = ric.assemble(verbose=verbose)
        if atom_coords:
            ric.ak_set = set(atom_coords.keys())