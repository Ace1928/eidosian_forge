from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
class VoronoiNN(NearNeighbors):
    """
    Uses a Voronoi algorithm to determine near neighbors for each site in a
    structure.
    """

    def __init__(self, tol=0, targets=None, cutoff=13.0, allow_pathological=False, weight='solid_angle', extra_nn_info=True, compute_adj_neighbors=True):
        """
        Args:
            tol (float): tolerance parameter for near-neighbor finding. Faces that are
                smaller than `tol` fraction of the largest face are not included in the
                tessellation. (default: 0).
            targets (Element or list of Elements): target element(s).
            cutoff (float): cutoff radius in Angstrom to look for near-neighbor
                atoms. Defaults to 13.0.
            allow_pathological (bool): whether to allow infinite vertices in
                determination of Voronoi coordination.
            weight (string) - Statistic used to weigh neighbors (see the statistics
                available in get_voronoi_polyhedra)
            extra_nn_info (bool) - Add all polyhedron info to `get_nn_info`
            compute_adj_neighbors (bool) - Whether to compute which neighbors are
                adjacent. Turn off for faster performance.
        """
        super().__init__()
        self.tol = tol
        self.cutoff = cutoff
        self.allow_pathological = allow_pathological
        self.targets = targets
        self.weight = weight
        self.extra_nn_info = extra_nn_info
        self.compute_adj_neighbors = compute_adj_neighbors

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return False

    def get_voronoi_polyhedra(self, structure: Structure, n: int):
        """
        Gives a weighted polyhedra around a site.

        See ref: A Proposed Rigorous Definition of Coordination Number,
        M. O'Keeffe, Acta Cryst. (1979). A35, 772-775

        Args:
            structure (Structure): structure for which to evaluate the
                coordination environment.
            n (int): site index.

        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """
        targets = structure.elements if self.targets is None else self.targets
        center = structure[n]
        corners = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        d_corners = [np.linalg.norm(structure.lattice.get_cartesian_coords(c)) for c in corners]
        max_cutoff = max(d_corners) + 0.01
        while True:
            try:
                neighbors = structure.get_sites_in_sphere(center.coords, self.cutoff)
                neighbors = [ngbr[0] for ngbr in sorted(neighbors, key=lambda s: s[1])]
                qvoronoi_input = [site.coords for site in neighbors]
                voro = Voronoi(qvoronoi_input)
                cell_info = self._extract_cell_info(0, neighbors, targets, voro, self.compute_adj_neighbors)
                break
            except RuntimeError as exc:
                if self.cutoff >= max_cutoff:
                    if exc.args and 'vertex' in exc.args[0]:
                        raise exc
                    raise RuntimeError('Error in Voronoi neighbor finding; max cutoff exceeded')
                self.cutoff = min(self.cutoff * 2, max_cutoff + 0.001)
        return cell_info

    def get_all_voronoi_polyhedra(self, structure: Structure):
        """Get the Voronoi polyhedra for all site in a simulation cell.

        Args:
            structure (Structure): Structure to be evaluated

        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """
        if len(structure) == 1:
            return [self.get_voronoi_polyhedra(structure, 0)]
        targets = structure.elements if self.targets is None else self.targets
        sites = [x.to_unit_cell() for x in structure]
        indices = [(idx, 0, 0, 0) for idx in range(len(structure))]
        all_neighs = structure.get_all_neighbors(self.cutoff, include_index=True, include_image=True)
        for neighs in all_neighs:
            sites.extend([x[0] for x in neighs])
            indices.extend([(x[2],) + x[3] for x in neighs])
        indices = np.array(indices, dtype=int)
        indices, uniq_inds = np.unique(indices, return_index=True, axis=0)
        sites = [sites[idx] for idx in uniq_inds]
        root_images, = np.nonzero(np.abs(indices[:, 1:]).max(axis=1) == 0)
        del indices
        qvoronoi_input = [s.coords for s in sites]
        voro = Voronoi(qvoronoi_input)
        return [self._extract_cell_info(idx, sites, targets, voro, self.compute_adj_neighbors) for idx in root_images.tolist()]

    def _extract_cell_info(self, site_idx, sites, targets, voro, compute_adj_neighbors=False):
        """Get the information about a certain atom from the results of a tessellation.

        Args:
            site_idx (int) - Index of the atom in question
            sites ([Site]) - List of all sites in the tessellation
            targets ([Element]) - Target elements
            voro - Output of qvoronoi
            compute_adj_neighbors (boolean) - Whether to compute which neighbors are adjacent

        Returns:
            A dict of sites sharing a common Voronoi facet. Key is facet id
             (not useful) and values are dictionaries containing statistics
             about the facet:
                - site: Pymatgen site
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
                - adj_neighbors - Facet id's for the adjacent neighbors
        """
        all_vertices = voro.vertices
        center_coords = sites[site_idx].coords
        results = {}
        for nn, vind in voro.ridge_dict.items():
            if site_idx in nn:
                other_site = nn[0] if nn[1] == site_idx else nn[1]
                if -1 in vind:
                    if self.allow_pathological:
                        continue
                    raise RuntimeError('This structure is pathological, infinite vertex in the Voronoi construction')
                facets = [all_vertices[idx] for idx in vind]
                angle = solid_angle(center_coords, facets)
                volume = 0
                for j, k in zip(vind[1:], vind[2:]):
                    volume += vol_tetra(center_coords, all_vertices[vind[0]], all_vertices[j], all_vertices[k])
                face_dist = np.linalg.norm(center_coords - sites[other_site].coords) / 2
                face_area = 3 * volume / face_dist
                normal = np.subtract(sites[other_site].coords, center_coords)
                normal /= np.linalg.norm(normal)
                results[other_site] = {'site': sites[other_site], 'normal': normal, 'solid_angle': angle, 'volume': volume, 'face_dist': face_dist, 'area': face_area, 'n_verts': len(vind)}
                if compute_adj_neighbors:
                    results[other_site]['verts'] = vind
        if len(results) == 0:
            raise ValueError('No Voronoi neighbors found for site - try increasing cutoff')
        result_weighted = {}
        for nn_index, nn_stats in results.items():
            nn = nn_stats['site']
            if nn.is_ordered:
                if nn.specie in targets:
                    result_weighted[nn_index] = nn_stats
            else:
                for disordered_sp in nn.species:
                    if disordered_sp in targets:
                        result_weighted[nn_index] = nn_stats
        if compute_adj_neighbors:
            adj_neighbors = {idx: [] for idx in result_weighted}
            for a_ind, a_nn_info in result_weighted.items():
                a_verts = set(a_nn_info['verts'])
                for b_ind, b_nninfo in result_weighted.items():
                    if b_ind > a_ind:
                        continue
                    if len(a_verts.intersection(b_nninfo['verts'])) == 2:
                        adj_neighbors[a_ind].append(b_ind)
                        adj_neighbors[b_ind].append(a_ind)
            for key, neighbors in adj_neighbors.items():
                result_weighted[key]['adj_neighbors'] = neighbors
        return result_weighted

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure
        using Voronoi decomposition.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        nns = self.get_voronoi_polyhedra(structure, n)
        return self._extract_nn_info(structure, nns)

    def get_all_nn_info(self, structure: Structure):
        """
        Args:
            structure (Structure): input structure.

        Returns:
            All nn info for all sites.
        """
        all_voro_cells = self.get_all_voronoi_polyhedra(structure)
        return [self._extract_nn_info(structure, cell) for cell in all_voro_cells]

    def _extract_nn_info(self, structure: Structure, nns):
        """Given Voronoi NNs, extract the NN info in the form needed by NearestNeighbors.

        Args:
            structure (Structure): Structure being evaluated
            nns ([dicts]): Nearest neighbor information for a structure

        Returns:
            list[tuple[PeriodicSite, np.ndarray, float]]: tuples of the form
                (site, image, weight). See nn_info.
        """
        targets = structure.elements if self.targets is None else self.targets
        siw = []
        max_weight = max((nn[self.weight] for nn in nns.values()))
        for nstats in nns.values():
            site = nstats['site']
            if nstats[self.weight] > self.tol * max_weight and _is_in_targets(site, targets):
                nn_info = {'site': site, 'image': self._get_image(structure, site), 'weight': nstats[self.weight] / max_weight, 'site_index': self._get_original_site(structure, site)}
                if self.extra_nn_info:
                    poly_info = nstats
                    del poly_info['site']
                    nn_info['poly_info'] = poly_info
                siw.append(nn_info)
        return siw