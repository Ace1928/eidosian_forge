from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
def types_of_coordination_environments(self, anonymous: bool=False) -> list[str]:
    """
        Extract information on the different co-ordination environments
        present in the graph.

        Args:
            anonymous: if anonymous, will replace specie names with A, B, C, etc.

        Returns:
            List of coordination environments, e.g. {'Mo-S(6)', 'S-Mo(3)'}
        """
    motifs = set()
    for idx, site in enumerate(self.structure):
        centre_sp = site.species_string
        connected_sites = self.get_connected_sites(idx)
        connected_species = [connected_site.site.species_string for connected_site in connected_sites]
        sp_counts = []
        for sp in set(connected_species):
            count = connected_species.count(sp)
            sp_counts.append((count, sp))
        sp_counts = sorted(sp_counts, reverse=True)
        if anonymous:
            mapping = {centre_sp: 'A'}
            available_letters = [chr(66 + idx) for idx in range(25)]
            for label in sp_counts:
                sp = label[1]
                if sp not in mapping:
                    mapping[sp] = available_letters.pop(0)
            centre_sp = 'A'
            sp_counts = [(label[0], mapping[label[1]]) for label in sp_counts]
        labels = [f'{label[1]}({label[0]})' for label in sp_counts]
        motif = f'{centre_sp}-{','.join(labels)}'
        motifs.add(motif)
    return sorted(set(motifs))