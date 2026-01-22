from __future__ import annotations
import logging
import warnings
import networkx as nx
from monty.json import MSONable
from pymatgen.analysis.fragmenter import open_ring
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor

        Build a new entry for bond dissociation that will be returned to the user.

        Args:
            frags (list): Fragments involved in the bond dissociation.
            bonds (list): Bonds broken in the dissociation process.

        Returns:
            list: Formatted bond dissociation entries.
        