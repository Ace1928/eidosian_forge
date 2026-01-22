from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def _evaluate_ce(self, lowerlimit, upperlimit, only_bonds_to=None, additional_condition: int=0, perc_strength_icohp: float=0.15, adapt_extremum_to_add_cond: bool=False) -> None:
    """
        Args:
            lowerlimit: lower limit which determines the ICOHPs that are considered for the determination of the
            neighbors
            upperlimit: upper limit which determines the ICOHPs that are considered for the determination of the
            neighbors
            only_bonds_to: restricts the types of bonds that will be considered
            additional_condition: Additional condition for the evaluation
            perc_strength_icohp: will be used to determine how strong the ICOHPs (percentage*strongest ICOHP) will be
            that are still considered for the evaluation
            adapt_extremum_to_add_cond: will recalculate the limit based on the bonding type and not on the overall
            extremum.
        """
    if lowerlimit is None and upperlimit is None:
        lowerlimit, upperlimit = self._get_limit_from_extremum(self.Icohpcollection, percentage=perc_strength_icohp, adapt_extremum_to_add_cond=adapt_extremum_to_add_cond, additional_condition=additional_condition)
    elif upperlimit is None or lowerlimit is None:
        raise ValueError('Please give two limits or leave them both at None')
    list_icohps, list_keys, list_lengths, list_neighisite, list_neighsite, list_coords = self._find_environments(additional_condition, lowerlimit, upperlimit, only_bonds_to)
    self.list_icohps = list_icohps
    self.list_lengths = list_lengths
    self.list_keys = list_keys
    self.list_neighsite = list_neighsite
    self.list_neighisite = list_neighisite
    self.list_coords = list_coords
    if self.add_additional_data_sg:
        self.sg_list = [[{'site': neighbor, 'image': tuple((int(round(i)) for i in neighbor.frac_coords - self.structure[next((isite for isite, site in enumerate(self.structure) if neighbor.is_periodic_image(site)))].frac_coords)), 'weight': 1, 'edge_properties': {'ICOHP': self.list_icohps[ineighbors][ineighbor], 'bond_length': self.list_lengths[ineighbors][ineighbor], 'bond_label': self.list_keys[ineighbors][ineighbor], self.id_blist_sg1.upper(): self.bonding_list_1.icohpcollection.get_icohp_by_label(self.list_keys[ineighbors][ineighbor]), self.id_blist_sg2.upper(): self.bonding_list_2.icohpcollection.get_icohp_by_label(self.list_keys[ineighbors][ineighbor])}, 'site_index': next((isite for isite, site in enumerate(self.structure) if neighbor.is_periodic_image(site)))} for ineighbor, neighbor in enumerate(neighbors)] for ineighbors, neighbors in enumerate(self.list_neighsite)]
    else:
        self.sg_list = [[{'site': neighbor, 'image': tuple((int(round(i)) for i in neighbor.frac_coords - self.structure[next((isite for isite, site in enumerate(self.structure) if neighbor.is_periodic_image(site)))].frac_coords)), 'weight': 1, 'edge_properties': {'ICOHP': self.list_icohps[ineighbors][ineighbor], 'bond_length': self.list_lengths[ineighbors][ineighbor], 'bond_label': self.list_keys[ineighbors][ineighbor]}, 'site_index': next((isite for isite, site in enumerate(self.structure) if neighbor.is_periodic_image(site)))} for ineighbor, neighbor in enumerate(neighbors)] for ineighbors, neighbors in enumerate(self.list_neighsite)]