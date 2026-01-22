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
class IsomorphismMolAtomMapper(AbstractMolAtomMapper):
    """Pair atoms by isomorphism permutations in the OpenBabel::OBAlign class."""

    def uniform_labels(self, mol1, mol2):
        """
        Pair the geometrically equivalent atoms of the molecules.
        Calculate RMSD on all possible isomorphism mappings and return mapping
        with the least RMSD.

        Args:
            mol1: First molecule. OpenBabel OBMol or pymatgen Molecule object.
            mol2: Second molecule. OpenBabel OBMol or pymatgen Molecule object.

        Returns:
            tuple[list1, list2]: if uniform atom order is found. list1 and list2
                are for mol1 and mol2, respectively. Their length equal
                to the number of atoms. They represents the uniform atom order
                of the two molecules. The value of each element is the original
                atom index in mol1 or mol2 of the current atom in uniform atom order.
                (None, None) if uniform atom is not available.
        """
        ob_mol1 = BabelMolAdaptor(mol1).openbabel_mol
        ob_mol2 = BabelMolAdaptor(mol2).openbabel_mol
        h1 = self.get_molecule_hash(ob_mol1)
        h2 = self.get_molecule_hash(ob_mol2)
        if h1 != h2:
            return (None, None)
        query = openbabel.CompileMoleculeQuery(ob_mol1)
        iso_mapper = openbabel.OBIsomorphismMapper.GetInstance(query)
        isomorph = openbabel.vvpairUIntUInt()
        iso_mapper.MapAll(ob_mol2, isomorph)
        sorted_isomorph = [sorted(x, key=lambda morp: morp[0]) for x in isomorph]
        label2_list = tuple((tuple((p[1] + 1 for p in x)) for x in sorted_isomorph))
        vmol1 = ob_mol1
        aligner = openbabel.OBAlign(True, False)
        aligner.SetRefMol(vmol1)
        least_rmsd = float('Inf')
        best_label2 = None
        label1 = list(range(1, ob_mol1.NumAtoms() + 1))
        elements1 = InchiMolAtomMapper._get_elements(vmol1, label1)
        for label2 in label2_list:
            elements2 = InchiMolAtomMapper._get_elements(ob_mol2, label2)
            if elements1 != elements2:
                continue
            vmol2 = openbabel.OBMol()
            for idx in label2:
                vmol2.AddAtom(ob_mol2.GetAtom(idx))
            aligner.SetTargetMol(vmol2)
            aligner.Align()
            rmsd = aligner.GetRMSD()
            if rmsd < least_rmsd:
                least_rmsd = rmsd
                best_label2 = copy.copy(label2)
        return (label1, best_label2)

    def get_molecule_hash(self, mol):
        """Return inchi as molecular hash."""
        ob_conv = openbabel.OBConversion()
        ob_conv.SetOutFormat('inchi')
        ob_conv.AddOption('X', openbabel.OBConversion.OUTOPTIONS, 'DoNotAddH')
        inchi_text = ob_conv.WriteString(mol)
        match = re.search('InChI=(?P<inchi>.+)\\n', inchi_text)
        return match.group('inchi')

    def as_dict(self):
        """
        Returns:
            JSON-able dict.
        """
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            IsomorphismMolAtomMapper
        """
        return cls()