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
class NearNeighbors:
    """
    Base class to determine near neighbors that typically include nearest
    neighbors and others that are within some tolerable distance.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self) -> int:
        return len(self.__dict__.items())

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        raise NotImplementedError('structures_allowed is not defined!')

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        raise NotImplementedError('molecules_allowed is not defined!')

    @property
    def extend_structure_molecules(self) -> bool:
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed is False.
        """
        raise NotImplementedError('extend_structures_molecule is not defined!')

    def get_cn(self, structure: Structure, n: int, use_weights: bool=False, on_disorder: on_disorder_options='take_majority_strict') -> float:
        """
        Get coordination number, CN, of site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True) to use weights for computing the coordination
                number or not (False, default: each coordinated site has equal weight).
            on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
                What to do when encountering a disordered structure. 'error' will raise ValueError.
                'take_majority_strict' will use the majority specie on each site and raise
                ValueError if no majority exists. 'take_max_species' will use the first max specie
                on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
                will raise ValueError, while 'take_majority_drop' ignores this site altogether and
                'take_max_species' will use Fe as the site specie.

        Returns:
            cn (float): coordination number.
        """
        structure = _handle_disorder(structure, on_disorder)
        siw = self.get_nn_info(structure, n)
        return sum((e['weight'] for e in siw)) if use_weights else len(siw)

    def get_cn_dict(self, structure: Structure, n: int, use_weights: bool=False):
        """
        Get coordination number, CN, of each element bonded to site with index n in structure.

        Args:
            structure (Structure): input structure
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).

        Returns:
            cn (dict): dictionary of CN of each element bonded to site
        """
        siw = self.get_nn_info(structure, n)
        cn_dict = {}
        for idx in siw:
            site_element = idx['site'].species_string
            if site_element not in cn_dict:
                if use_weights:
                    cn_dict[site_element] = idx['weight']
                else:
                    cn_dict[site_element] = 1
            elif use_weights:
                cn_dict[site_element] += idx['weight']
            else:
                cn_dict[site_element] += 1
        return cn_dict

    def get_nn(self, structure: Structure, n: int):
        """
        Get near neighbors of site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site in structure for which to determine
                    neighbors.

        Returns:
            sites (list of Site objects): near neighbors.
        """
        return [e['site'] for e in self.get_nn_info(structure, n)]

    def get_weights_of_nn_sites(self, structure: Structure, n: int):
        """
        Get weight associated with each near neighbor of site with
        index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine the weights.

        Returns:
            weights (list of floats): near-neighbor weights.
        """
        return [e['weight'] for e in self.get_nn_info(structure, n)]

    def get_nn_images(self, structure: Structure, n: int):
        """
        Get image location of all near neighbors of site with index n in
        structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine the image
                location of near neighbors.

        Returns:
            images (list of 3D integer array): image locations of
                near neighbors.
        """
        return [e['image'] for e in self.get_nn_info(structure, n)]

    def get_nn_info(self, structure: Structure, n: int) -> list[dict]:
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor
                information.

        Returns:
            siw (list[dict]): each dictionary provides information
                about a single near neighbor, where key 'site' gives access to the
                corresponding Site object, 'image' gives the image location, and
                'weight' provides the weight that a given near-neighbor site contributes
                to the coordination number (1 or smaller), 'site_index' gives index of
                the corresponding site in the original structure.
        """
        raise NotImplementedError('get_nn_info(structure, n) is not defined!')

    def get_all_nn_info(self, structure: Structure):
        """Get a listing of all neighbors for all sites in a structure.

        Args:
            structure (Structure): Input structure
        Returns:
            List of NN site information for each site in the structure. Each
                entry has the same format as `get_nn_info`
        """
        return [self.get_nn_info(structure, n) for n in range(len(structure))]

    def get_nn_shell_info(self, structure: Structure, site_idx, shell):
        """Get a certain nearest neighbor shell for a certain site.

        Determines all non-backtracking paths through the neighbor network
        computed by `get_nn_info`. The weight is determined by multiplying
        the weight of the neighbor at each hop through the network. For
        example, a 2nd-nearest-neighbor that has a weight of 1 from its
        1st-nearest-neighbor and weight 0.5 from the original site will
        be assigned a weight of 0.5.

        As this calculation may involve computing the nearest neighbors of
        atoms multiple times, the calculation starts by computing all of the
        neighbor info and then calling `_get_nn_shell_info`. If you are likely
        to call this method for more than one site, consider calling `get_all_nn`
        first and then calling this protected method yourself.

        Args:
            structure (Structure): Input structure
            site_idx (int): index of site for which to determine neighbor
                information.
            shell (int): Which neighbor shell to retrieve (1 == 1st NN shell)

        Returns:
            list of dictionaries. Each entry in the list is information about
                a certain neighbor in the structure, in the same format as
                `get_nn_info`.
        """
        all_nn_info = self.get_all_nn_info(structure)
        sites = self._get_nn_shell_info(structure, all_nn_info, site_idx, shell)
        output = []
        for info in sites:
            orig_site = structure[info['site_index']]
            info['site'] = PeriodicSite(orig_site.species, np.add(orig_site.frac_coords, info['image']), structure.lattice, properties=orig_site.properties)
            output.append(info)
        return output

    def _get_nn_shell_info(self, structure, all_nn_info, site_idx, shell, _previous_steps=frozenset(), _cur_image=(0, 0, 0)):
        """Private method for computing the neighbor shell information.

        Args:
            structure (Structure) - Structure being assessed
            all_nn_info ([[dict]]) - Results from `get_all_nn_info`
            site_idx (int) - index of site for which to determine neighbor
                information.
            shell (int) - Which neighbor shell to retrieve (1 == 1st NN shell)
            _previous_steps ({(site_idx, image}) - Internal use only: Set of
                sites that have already been traversed.
            _cur_image (tuple) - Internal use only Image coordinates of current atom

        Returns:
            list of dictionaries. Each entry in the list is information about
                a certain neighbor in the structure, in the same format as
                `get_nn_info`. Does not update the site positions
        """
        if shell <= 0:
            raise ValueError('Shell must be positive')
        _previous_steps = _previous_steps | {(site_idx, _cur_image)}
        possible_steps = list(all_nn_info[site_idx])
        for idx, step in enumerate(possible_steps):
            step = dict(step)
            step['image'] = tuple(np.add(step['image'], _cur_image).tolist())
            possible_steps[idx] = step
        allowed_steps = [x for x in possible_steps if (x['site_index'], x['image']) not in _previous_steps]
        if shell == 1:
            return allowed_steps
        terminal_neighbors = [self._get_nn_shell_info(structure, all_nn_info, x['site_index'], shell - 1, _previous_steps, x['image']) for x in allowed_steps]
        all_sites = {}
        for first_site, term_sites in zip(allowed_steps, terminal_neighbors):
            for term_site in term_sites:
                key = (term_site['site_index'], tuple(term_site['image']))
                term_site['weight'] *= first_site['weight']
                value = all_sites.get(key)
                if value is not None:
                    value['weight'] += term_site['weight']
                else:
                    value = term_site
                all_sites[key] = value
        return list(all_sites.values())

    @staticmethod
    def _get_image(structure: Structure, site: Site) -> tuple[int, int, int]:
        """Private convenience method for get_nn_info,
        gives lattice image from provided PeriodicSite and Structure.

        Image is defined as displacement from original site in structure to a given site.
        i.e. if structure has a site at (-0.1, 1.0, 0.3), then (0.9, 0, 2.3) -> jimage = (1, -1, 2).
        Note that this method takes O(number of sites) due to searching an original site.

        Args:
            structure (Structure): Structure Object
            site (Site): PeriodicSite Object

        Returns:
            tuple[int, int , int] Lattice image
        """
        if isinstance(site, PeriodicNeighbor):
            return site.image
        original_site = structure[NearNeighbors._get_original_site(structure, site)]
        image = np.around(np.subtract(site.frac_coords, original_site.frac_coords))
        return tuple(image.astype(int))

    @staticmethod
    def _get_original_site(structure: Structure, site: Site) -> int:
        """Private convenience method for get_nn_info,
        gives original site index from ProvidedPeriodicSite.
        """
        if isinstance(site, PeriodicNeighbor):
            return site.index
        if isinstance(structure, (IStructure, Structure)):
            for idx, struc_site in enumerate(structure):
                if site.is_periodic_image(struc_site):
                    return idx
        else:
            for idx, struc_site in enumerate(structure):
                if site == struc_site:
                    return idx
        raise ValueError('Site not found in structure')

    def get_bonded_structure(self, structure: Structure, decorate: bool=False, weights: bool=True, edge_properties: bool=False, on_disorder: on_disorder_options='take_majority_strict') -> StructureGraph | MoleculeGraph:
        """
        Obtain a StructureGraph object using this NearNeighbor
        class. Requires the optional dependency networkx
        (pip install networkx).

        Args:
            structure: Structure object.
            decorate (bool): whether to annotate site properties with order parameters using neighbors
                determined by this NearNeighbor class
            weights (bool): whether to include edge weights from NearNeighbor class in StructureGraph
            edge_properties (bool) whether to include further edge properties from NearNeighbor class in StructureGraph
            on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
                What to do when encountering a disordered structure. 'error' will raise ValueError.
                'take_majority_strict' will use the majority specie on each site and raise
                ValueError if no majority exists. 'take_max_species' will use the first max specie
                on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
                will raise ValueError, while 'take_majority_drop' ignores this site altogether and
                'take_max_species' will use Fe as the site specie.

        Returns:
            StructureGraph: object from pymatgen.analysis.graphs
        """
        structure = _handle_disorder(structure, on_disorder)
        if decorate:
            order_parameters = [self.get_local_order_parameters(structure, n) for n in range(len(structure))]
            structure.add_site_property('order_parameters', order_parameters)
        struct_graph = StructureGraph.from_local_env_strategy(structure, self, weights=weights, edge_properties=edge_properties)
        struct_graph.set_node_attributes()
        return struct_graph

    def get_local_order_parameters(self, structure: Structure, n: int):
        """
        Calculate those local structure order parameters for
        the given site whose ideal CN corresponds to the
        underlying motif (e.g., CN=4, then calculate the
        square planar, tetrahedral, see-saw-like,
        rectangular see-saw-like order parameters).

        Args:
            structure: Structure object
            n (int): site index.

        Returns:
            dict[str, float]: A dict of order parameters (values) and the
                underlying motif type (keys; for example, tetrahedral).
        """
        cn = self.get_cn(structure, n)
        int_cn = [int(k_cn) for k_cn in cn_opt_params]
        if cn in int_cn:
            names = list(cn_opt_params[cn])
            types = []
            params = []
            for name in names:
                types.append(cn_opt_params[cn][name][0])
                tmp = cn_opt_params[cn][name][1] if len(cn_opt_params[cn][name]) > 1 else None
                params.append(tmp)
            lsops = LocalStructOrderParams(types, parameters=params)
            sites = [structure[n], *self.get_nn(structure, n)]
            lostop_vals = lsops.get_order_parameters(sites, 0, indices_neighs=list(range(1, cn + 1)))
            dct = {}
            for idx, lsop in enumerate(lostop_vals):
                dct[names[idx]] = lsop
            return dct
        return None