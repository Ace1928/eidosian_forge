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
class MoleculeMatcher(MSONable):
    """Class to match molecules and identify whether molecules are the same."""

    @requires(openbabel, 'BabelMolAdaptor requires openbabel to be installed with Python bindings. Please get it at http://openbabel.org (version >=3.0.0).')
    def __init__(self, tolerance: float=0.01, mapper=None) -> None:
        """
        Args:
            tolerance (float): RMSD difference threshold whether two molecules are
                different
            mapper (AbstractMolAtomMapper): MolAtomMapper object that is able to map the atoms of two
                molecule to uniform order.
        """
        self._tolerance = tolerance
        self._mapper = mapper or InchiMolAtomMapper()

    def fit(self, mol1, mol2):
        """
        Fit two molecules.

        Args:
            mol1: First molecule. OpenBabel OBMol or pymatgen Molecule object
            mol2: Second molecule. OpenBabel OBMol or pymatgen Molecule object

        Returns:
            A boolean value indicates whether two molecules are the same.
        """
        return self.get_rmsd(mol1, mol2) < self._tolerance

    def get_rmsd(self, mol1, mol2):
        """
        Get RMSD between two molecule with arbitrary atom order.

        Returns:
            RMSD if topology of the two molecules are the same
            Infinite if  the topology is different
        """
        label1, label2 = self._mapper.uniform_labels(mol1, mol2)
        if label1 is None or label2 is None:
            return float('Inf')
        return self._calc_rms(mol1, mol2, label1, label2)

    @staticmethod
    def _calc_rms(mol1, mol2, clabel1, clabel2):
        """
        Calculate the RMSD.

        Args:
            mol1: The first molecule. OpenBabel OBMol or pymatgen Molecule
                object
            mol2: The second molecule. OpenBabel OBMol or pymatgen Molecule
                object
            clabel1: The atom indices that can reorder the first molecule to
                uniform atom order
            clabel1: The atom indices that can reorder the second molecule to
                uniform atom order

        Returns:
            The RMSD.
        """
        ob_mol1 = BabelMolAdaptor(mol1).openbabel_mol
        ob_mol2 = BabelMolAdaptor(mol2).openbabel_mol
        cmol1 = openbabel.OBMol()
        for idx in clabel1:
            oa1 = ob_mol1.GetAtom(idx)
            a1 = cmol1.NewAtom()
            a1.SetAtomicNum(oa1.GetAtomicNum())
            a1.SetVector(oa1.GetVector())
        cmol2 = openbabel.OBMol()
        for idx in clabel2:
            oa2 = ob_mol2.GetAtom(idx)
            a2 = cmol2.NewAtom()
            a2.SetAtomicNum(oa2.GetAtomicNum())
            a2.SetVector(oa2.GetVector())
        aligner = openbabel.OBAlign(True, False)
        aligner.SetRefMol(cmol1)
        aligner.SetTargetMol(cmol2)
        aligner.Align()
        return aligner.GetRMSD()

    def group_molecules(self, mol_list):
        """
        Group molecules by structural equality.

        Args:
            mol_list: List of OpenBabel OBMol or pymatgen objects

        Returns:
            A list of lists of matched molecules
            Assumption: if s1=s2 and s2=s3, then s1=s3
            This may not be true for small tolerances.
        """
        mol_hash = [(idx, self._mapper.get_molecule_hash(mol)) for idx, mol in enumerate(mol_list)]
        mol_hash.sort(key=lambda x: x[1])
        raw_groups = tuple((tuple((m[0] for m in g)) for k, g in itertools.groupby(mol_hash, key=lambda x: x[1])))
        group_indices = []
        for rg in raw_groups:
            mol_eq_test = [(p[0], p[1], self.fit(mol_list[p[0]], mol_list[p[1]])) for p in itertools.combinations(sorted(rg), 2)]
            mol_eq = {(p[0], p[1]) for p in mol_eq_test if p[2]}
            not_alone_mols = set(itertools.chain.from_iterable(mol_eq))
            alone_mols = set(rg) - not_alone_mols
            group_indices.extend([[m] for m in alone_mols])
            while len(not_alone_mols) > 0:
                current_group = {not_alone_mols.pop()}
                while len(not_alone_mols) > 0:
                    candidate_pairs = {tuple(sorted(p)) for p in itertools.product(current_group, not_alone_mols)}
                    mutual_pairs = candidate_pairs & mol_eq
                    if len(mutual_pairs) == 0:
                        break
                    mutual_mols = set(itertools.chain.from_iterable(mutual_pairs))
                    current_group |= mutual_mols
                    not_alone_mols -= mutual_mols
                group_indices.append(sorted(current_group))
        group_indices.sort(key=lambda x: (len(x), -x[0]), reverse=True)
        return [[mol_list[idx] for idx in g] for g in group_indices]

    def as_dict(self):
        """
        Returns:
            MSONable dict.
        """
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'tolerance': self._tolerance, 'mapper': self._mapper.as_dict()}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            MoleculeMatcher
        """
        return cls(tolerance=dct['tolerance'], mapper=AbstractMolAtomMapper.from_dict(dct['mapper']))