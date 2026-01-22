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
class AbstractMolAtomMapper(MSONable, abc.ABC):
    """
    Abstract molecular atom order mapping class. A mapping will be able to
    find the uniform atom order of two molecules that can pair the
    geometrically equivalent atoms.
    """

    @abc.abstractmethod
    def uniform_labels(self, mol1, mol2):
        """
        Pair the geometrically equivalent atoms of the molecules.

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

    @abc.abstractmethod
    def get_molecule_hash(self, mol):
        """
        Defines a hash for molecules. This allows molecules to be grouped
        efficiently for comparison.

        Args:
            mol: The molecule. OpenBabel OBMol or pymatgen Molecule object

        Returns:
            A hashable object. Examples can be string formulas, etc.
        """

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            AbstractMolAtomMapper
        """
        for trans_modules in ['molecule_matcher']:
            level = 0
            mod = __import__(f'pymatgen.analysis.{trans_modules}', globals(), locals(), [dct['@class']], level)
            if hasattr(mod, dct['@class']):
                class_proxy = getattr(mod, dct['@class'])
                return class_proxy.from_dict(dct)
        raise ValueError('Invalid Comparator dict')