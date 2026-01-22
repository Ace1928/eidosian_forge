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
def assemble_residues(self, verbose: bool=False) -> None:
    """Generate atom coords from internal coords (vectorised).

        This is the 'Numpy parallel' version of :meth:`.assemble_residues_ser`.

        Starting with dihedra already formed by :meth:`.init_atom_coords`, transform
        each from dihedron local coordinate space into protein chain coordinate
        space.  Iterate until all dependencies satisfied.

        Does not update :data:`dCoordSpace` as :meth:`assemble_residues_ser`
        does.  Call :meth:`.update_dCoordSpace` if needed.  Faster to do in
        single operation once all atom coordinates finished.

        :param bool verbose: default False.
            Report number of iterations to compute changed dihedra

        generates:
            self.dSet: AAsiz x dihedraLen x 4
                maps atoms in dihedra to atomArray
            self.dSetValid : [dihedraLen][4] of bool
                map of valid atoms into dihedra to detect 3 or 4 atoms valid

        Output coordinates written to :data:`atomArray`.  Biopython
        :class:`Bio.PDB.Atom` coordinates are a view on this data.
        """
    a2da_map = self.a2da_map
    a2d_map = self.a2d_map
    d2a_map = self.d2a_map
    atomArray = self.atomArray
    atomArrayValid = self.atomArrayValid
    dAtoms = self.dAtoms
    dCoordSpace1 = self.dCoordSpace[1]
    dcsValid = self.dcsValid
    self.dSet = atomArray[a2da_map].reshape(-1, 4, 4)
    dSet = self.dSet
    self.dSetValid = atomArrayValid[a2da_map].reshape(-1, 4)
    dSetValid = self.dSetValid
    workSelector = (dSetValid == self._dihedraOK).all(axis=1)
    self.dcsValid[np.logical_not(workSelector)] = False
    dihedraWrk = None
    if verbose:
        dihedraWrk = workSelector.size - workSelector.sum()
    targ = IC_Chain._dihedraSelect
    workSelector = (dSetValid == targ).all(axis=1)
    loopCount = 0
    while np.any(workSelector):
        workNdxs = np.where(workSelector)
        workSet = dSet[workSelector]
        updateMap = d2a_map[workNdxs, 3][0]
        if np.all(dcsValid[workSelector]):
            cspace = dCoordSpace1[workSelector]
        else:
            cspace = multi_coord_space(workSet, np.sum(workSelector), True)[1]
        initCoords = dAtoms[workSelector].reshape(-1, 4, 4)
        atomArray[updateMap] = np.einsum('ijk,ik->ij', cspace, initCoords[:, 3])
        atomArrayValid[updateMap] = True
        workSelector[:] = False
        for a in updateMap:
            dSet[a2d_map[a]] = atomArray[a]
            adlist = a2d_map[a]
            for d in adlist[0]:
                dvalid = atomArrayValid[d2a_map[d]]
                workSelector[d] = (dvalid == targ).all()
        loopCount += 1
    if verbose:
        cid = self.chain.full_id
        print(f'{cid[0]} {cid[2]} coordinates for {dihedraWrk} dihedra updated in {loopCount} iterations')