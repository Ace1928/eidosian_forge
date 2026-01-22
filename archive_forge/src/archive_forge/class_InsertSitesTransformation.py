from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
class InsertSitesTransformation(AbstractTransformation):
    """This transformation substitutes certain sites with certain species."""

    def __init__(self, species, coords, coords_are_cartesian=False, validate_proximity=True):
        """
        Args:
            species: A list of species. e.g., ["Li", "Fe"]
            coords: A list of coords corresponding to those species. e.g.,
                [[0,0,0],[0.5,0.5,0.5]].
            coords_are_cartesian (bool): Set to True if coords are given in
                Cartesian coords. Defaults to False.
            validate_proximity (bool): Set to False if you do not wish to ensure
                that added sites are not too close to other sites. Defaults to True.
        """
        if len(species) != len(coords):
            raise ValueError('Species and coords must be the same length!')
        self.species = species
        self.coords = coords
        self.coords_are_cartesian = coords_are_cartesian
        self.validate_proximity = validate_proximity

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.

        Returns:
            A copy of structure with sites inserted.
        """
        struct = structure.copy()
        for idx, sp in enumerate(self.species):
            struct.insert(idx, sp, self.coords[idx], coords_are_cartesian=self.coords_are_cartesian, validate_proximity=self.validate_proximity)
        return struct.get_sorted_structure()

    def __repr__(self):
        return f'InsertSiteTransformation : species {self.species}, coords {self.coords}'

    @property
    def inverse(self):
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False