from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
@staticmethod
def _align_hydrogen_atoms(mol1, mol2, heavy_indices1, heavy_indices2):
    """
        Align the label of topologically identical atoms of second molecule
        towards first molecule.

        Args:
            mol1: First molecule. OpenBabel OBMol object
            mol2: Second molecule. OpenBabel OBMol object
            heavy_indices1: inchi label map of the first molecule
            heavy_indices2: label map of the second molecule

        Returns:
            corrected label map of all atoms of the second molecule
        """
    num_atoms = mol2.NumAtoms()
    all_atom = set(range(1, num_atoms + 1))
    hydrogen_atoms1 = all_atom - set(heavy_indices1)
    hydrogen_atoms2 = all_atom - set(heavy_indices2)
    label1 = heavy_indices1 + tuple(hydrogen_atoms1)
    label2 = heavy_indices2 + tuple(hydrogen_atoms2)
    cmol1 = openbabel.OBMol()
    for idx in label1:
        oa1 = mol1.GetAtom(idx)
        a1 = cmol1.NewAtom()
        a1.SetAtomicNum(oa1.GetAtomicNum())
        a1.SetVector(oa1.GetVector())
    cmol2 = openbabel.OBMol()
    for idx in label2:
        oa2 = mol2.GetAtom(idx)
        a2 = cmol2.NewAtom()
        a2.SetAtomicNum(oa2.GetAtomicNum())
        a2.SetVector(oa2.GetVector())
    aligner = openbabel.OBAlign(False, False)
    aligner.SetRefMol(cmol1)
    aligner.SetTargetMol(cmol2)
    aligner.Align()
    aligner.UpdateCoords(cmol2)
    hydrogen_label2 = []
    hydrogen_label1 = list(range(len(heavy_indices1) + 1, num_atoms + 1))
    for h2 in range(len(heavy_indices2) + 1, num_atoms + 1):
        distance = 99999.0
        idx = hydrogen_label1[0]
        a2 = cmol2.GetAtom(h2)
        for h1 in hydrogen_label1:
            a1 = cmol1.GetAtom(h1)
            dist = a1.GetDistance(a2)
            if dist < distance:
                distance = dist
                idx = h1
        hydrogen_label2.append(idx)
        hydrogen_label1.remove(idx)
    hydrogen_orig_idx2 = label2[len(heavy_indices2):]
    hydrogen_canon_orig_map2 = list(zip(hydrogen_label2, hydrogen_orig_idx2))
    hydrogen_canon_orig_map2.sort(key=lambda m: m[0])
    hydrogen_canon_indices2 = [x[1] for x in hydrogen_canon_orig_map2]
    canon_label1 = label1
    canon_label2 = heavy_indices2 + tuple(hydrogen_canon_indices2)
    return (canon_label1, canon_label2)