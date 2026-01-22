from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def _get_number_of_d_electrons(species: Species) -> float:
    """
        Get number of d electrons of a species.

        Args:
            species: Species object

        Returns:
            int: Number of d electrons.
        """
    elec = species.full_electronic_structure
    if len(elec) < 4 or elec[-1][1] != 's' or elec[-2][1] != 'd':
        raise AttributeError(f'Invalid element {species.symbol} for crystal field calculation.')
    n_electrons = int(elec[-1][2] + elec[-2][2] - species.oxi_state)
    if n_electrons < 0 or n_electrons > 10:
        raise AttributeError(f'Invalid oxidation state {species.oxi_state} for element {species.symbol}')
    return n_electrons