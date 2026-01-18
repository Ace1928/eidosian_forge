from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor

        Determine classes of functional groups present in a set.

        Args:
            groups: Set of functional groups.

        Returns:
            dict containing representations of the groups, the indices of
            where the group occurs in the MoleculeGraph, and how many of each
            type of group there is.
        