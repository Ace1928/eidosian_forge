from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
class AbstractGeometry:
    """Class used to describe a geometry (perfect or distorted)."""

    def __init__(self, central_site=None, bare_coords=None, centering_type='standard', include_central_site_in_centroid=False, optimization=None):
        """
        Constructor for the abstract geometry

        Args:
            central_site: Coordinates of the central site
            bare_coords: Coordinates of the neighbors of the central site
            centering_type: How to center the abstract geometry
            include_central_site_in_centroid: When the centering is on the centroid,
                the central site is included if this parameter is set to True.

        Raises:
            ValueError if the parameters are not consistent.
        """
        bcoords = np.array(bare_coords)
        self.bare_centre = np.array(central_site)
        self.bare_points_without_centre = bcoords
        self.bare_points_with_centre = np.array(central_site)
        self.bare_points_with_centre = np.concatenate(([self.bare_points_with_centre], bcoords))
        self.centroid_without_centre = np.mean(self.bare_points_without_centre, axis=0)
        self.centroid_with_centre = np.mean(self.bare_points_with_centre, axis=0)
        self._points_wcs_csc = self.bare_points_with_centre - self.bare_centre
        self._points_wocs_csc = self.bare_points_without_centre - self.bare_centre
        self._points_wcs_ctwcc = self.bare_points_with_centre - self.centroid_with_centre
        self._points_wocs_ctwcc = self.bare_points_without_centre - self.centroid_with_centre
        self._points_wcs_ctwocc = self.bare_points_with_centre - self.centroid_without_centre
        self._points_wocs_ctwocc = self.bare_points_without_centre - self.centroid_without_centre
        self.centering_type = centering_type
        self.include_central_site_in_centroid = include_central_site_in_centroid
        self.bare_central_site = np.array(central_site)
        if centering_type == 'standard':
            if len(bare_coords) < 5:
                if include_central_site_in_centroid:
                    raise ValueError('The center is the central site, no calculation of the centroid, variable include_central_site_in_centroid should be set to False')
                if central_site is None:
                    raise ValueError('Centering_type is central_site, the central site should be given')
                self.centre = np.array(central_site)
            else:
                total = np.sum(bcoords, axis=0)
                if include_central_site_in_centroid:
                    if central_site is None:
                        raise ValueError('The centroid includes the central site but no central site is given')
                    total += self.bare_centre
                    self.centre = total / (np.float64(len(bare_coords)) + 1.0)
                else:
                    self.centre = total / np.float64(len(bare_coords))
        elif centering_type == 'central_site':
            if include_central_site_in_centroid:
                raise ValueError('The center is the central site, no calculation of the centroid, variable include_central_site_in_centroid should be set to False')
            if central_site is None:
                raise ValueError('Centering_type is central_site, the central site should be given')
            self.centre = np.array(central_site)
        elif centering_type == 'centroid':
            total = np.sum(bcoords, axis=0)
            if include_central_site_in_centroid:
                if central_site is None:
                    raise ValueError('The centroid includes the central site but no central site is given')
                total += self.bare_centre
                self.centre = total / (np.float64(len(bare_coords)) + 1.0)
            else:
                self.centre = total / np.float64(len(bare_coords))
        self._bare_coords = self.bare_points_without_centre
        self._coords = self._bare_coords - self.centre
        self.central_site = self.bare_central_site - self.centre
        self.coords = self._coords
        self.bare_coords = self._bare_coords

    def __str__(self):
        """
        String representation of the AbstractGeometry

        Returns:
            str: String representation of the AbstractGeometry.
        """
        outs = [f'\nAbstract Geometry with {len(self.coords)} points :']
        for pp in self.coords:
            outs.append(f'  {pp}')
        if self.centering_type == 'standard':
            if self.include_central_site_in_centroid:
                outs.append(f'Points are referenced to the central site for coordination numbers < 5 and to the centroid (calculated with the central site) for coordination numbers >= 5 : {self.centre}\n')
            else:
                outs.append(f'Points are referenced to the central site for coordination numbers < 5 and to the centroid (calculated without the central site) for coordination numbers >= 5 : {self.centre}\n')
        elif self.centering_type == 'central_site':
            outs.append(f'Points are referenced to the central site : {self.centre}\n')
        elif self.centering_type == 'centroid':
            if self.include_central_site_in_centroid:
                outs.append(f'Points are referenced to the centroid (calculated with the central site) :\n  {self.centre}\n')
            else:
                outs.append(f'Points are referenced to the centroid (calculated without the central site) :\n  {self.centre}\n')
        return '\n'.join(outs)

    @classmethod
    def from_cg(cls, cg, centering_type='standard', include_central_site_in_centroid=False) -> Self:
        """
        Args:
            cg:
            centering_type:
            include_central_site_in_centroid:
        """
        central_site = cg.get_central_site()
        bare_coords = [np.array(pt, float) for pt in cg.points]
        return cls(central_site=central_site, bare_coords=bare_coords, centering_type=centering_type, include_central_site_in_centroid=include_central_site_in_centroid)

    def points_wcs_csc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wcs_csc
        return np.concatenate((self._points_wcs_csc[0:1], self._points_wocs_csc.take(permutation, axis=0)))

    def points_wocs_csc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wocs_csc
        return self._points_wocs_csc.take(permutation, axis=0)

    def points_wcs_ctwcc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wcs_ctwcc
        return np.concatenate((self._points_wcs_ctwcc[0:1], self._points_wocs_ctwcc.take(permutation, axis=0)))

    def points_wocs_ctwcc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wocs_ctwcc
        return self._points_wocs_ctwcc.take(permutation, axis=0)

    def points_wcs_ctwocc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wcs_ctwocc
        return np.concatenate((self._points_wcs_ctwocc[0:1], self._points_wocs_ctwocc.take(permutation, axis=0)))

    def points_wocs_ctwocc(self, permutation=None):
        """
        Args:
            permutation:
        """
        if permutation is None:
            return self._points_wocs_ctwocc
        return self._points_wocs_ctwocc.take(permutation, axis=0)

    @property
    def cn(self):
        """Coordination number"""
        return len(self.coords)

    @property
    def coordination_number(self):
        """Coordination number"""
        return len(self.coords)