from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def _get_graphs(cutoff, ordered_structures):
    """
        Generate graph representations of magnetic structures with nearest
        neighbor bonds. Right now this only works for MinimumDistanceNN.

        Args:
            cutoff (float): Cutoff in Angstrom for nearest neighbor search.
            ordered_structures (list): Structure objects.

        Returns:
            sgraphs (list): StructureGraph objects.
        """
    strategy = MinimumDistanceNN(cutoff=cutoff, get_all_sites=True) if cutoff else MinimumDistanceNN()
    return [StructureGraph.from_local_env_strategy(s, strategy=strategy) for s in ordered_structures]