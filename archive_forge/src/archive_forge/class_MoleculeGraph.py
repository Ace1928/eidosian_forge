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
class MoleculeGraph(MSONable):
    """
    This is a class for annotating a Molecule with
    bond information, stored in the form of a graph. A "bond" does
    not necessarily have to be a chemical bond, but can store any
    kind of information that connects two Sites.
    """

    def __init__(self, molecule, graph_data=None):
        """
        If constructing this class manually, use the `from_empty_graph`
        method or `from_local_env_strategy` method (using an algorithm
        provided by the `local_env` module, such as O'Keeffe).

        This class that contains connection information:
        relationships between sites represented by a Graph structure,
        and an associated structure object.

        This class uses the NetworkX package to store and operate
        on the graph itself, but contains a lot of helper methods
        to make associating a graph with a given molecule easier.

        Use cases for this include storing bonding information,
        NMR J-couplings, Heisenberg exchange parameters, etc.

        Args:
            molecule: Molecule object

            graph_data: dict containing graph information in
                dict format (not intended to be constructed manually,
                see as_dict method for format)
        """
        if isinstance(molecule, MoleculeGraph):
            graph_data = molecule.as_dict()['graphs']
        self.molecule = molecule
        self.graph = nx.readwrite.json_graph.adjacency_graph(graph_data)
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            for key in ('id', 'key'):
                data.pop(key, None)
            if 'to_jimage' in data:
                data['to_jimage'] = tuple(data['to_jimage'])
            if 'from_jimage' in data:
                data['from_jimage'] = tuple(data['from_jimage'])
        self.set_node_attributes()

    @classmethod
    def from_empty_graph(cls, molecule, name='bonds', edge_weight_name=None, edge_weight_units=None) -> Self:
        """
        Constructor for MoleculeGraph, returns a MoleculeGraph
        object with an empty graph (no edges, only nodes defined
        that correspond to Sites in Molecule).

        Args:
            molecule (Molecule):
            name (str): name of graph, e.g. "bonds"
            edge_weight_name (str): name of edge weights,
                e.g. "bond_length" or "exchange_constant"
            edge_weight_units (str): name of edge weight units
            e.g. "Å" or "eV"

        Returns:
            MoleculeGraph
        """
        if edge_weight_name and edge_weight_units is None:
            raise ValueError('Please specify units associated with your edge weights. Can be empty string if arbitrary or dimensionless.')
        graph = nx.MultiDiGraph(edge_weight_name=edge_weight_name, edge_weight_units=edge_weight_units, name=name)
        graph.add_nodes_from(range(len(molecule)))
        graph_data = json_graph.adjacency_data(graph)
        return cls(molecule, graph_data=graph_data)

    @classmethod
    @deprecated(from_empty_graph, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_empty_graph(cls, *args, **kwargs):
        return cls.from_empty_graph(*args, **kwargs)

    @classmethod
    def from_edges(cls, molecule: Molecule, edges: dict[tuple[int, int], None | dict]) -> Self:
        """
        Constructor for MoleculeGraph, using pre-existing or pre-defined edges
        with optional edge parameters.

        Args:
            molecule: Molecule object
            edges: dict representing the bonds of the functional
                group (format: {(u, v): props}, where props is a dictionary of
                properties, including weight. Props should be None if no
                additional properties are to be specified.

        Returns:
            A MoleculeGraph
        """
        mg = cls.from_empty_graph(molecule, name='bonds', edge_weight_name='weight', edge_weight_units='')
        for edge, props in edges.items():
            try:
                from_index = edge[0]
                to_index = edge[1]
            except TypeError:
                raise ValueError('Edges must be given as (from_index, to_index) tuples')
            if props is None:
                weight = None
            else:
                weight = props.pop('weight', None)
                if len(props.items()) == 0:
                    props = None
            nodes = mg.graph.nodes
            if not (from_index in nodes and to_index in nodes):
                raise ValueError('Edges cannot be added if nodes are not present in the graph. Please check your indices.')
            mg.add_edge(from_index, to_index, weight=weight, edge_properties=props)
        mg.set_node_attributes()
        return mg

    @classmethod
    @deprecated(from_edges, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_edges(cls, *args, **kwargs):
        return cls.from_edges(*args, **kwargs)

    @classmethod
    def from_local_env_strategy(cls, molecule, strategy) -> Self:
        """
        Constructor for MoleculeGraph, using a strategy
        from pymatgen.analysis.local_env.

            molecule: Molecule object
            strategy: an instance of a
                pymatgen.analysis.local_env.NearNeighbors object

        Returns:
            mg, a MoleculeGraph
        """
        if not strategy.molecules_allowed:
            raise ValueError(f'strategy={strategy!r} is not designed for use with molecules! Choose another strategy.')
        extend_structure = strategy.extend_structure_molecules
        mg = cls.from_empty_graph(molecule, name='bonds', edge_weight_name='weight', edge_weight_units='')
        coords = molecule.cart_coords
        if extend_structure:
            a = max(coords[:, 0]) - min(coords[:, 0]) + 100
            b = max(coords[:, 1]) - min(coords[:, 1]) + 100
            c = max(coords[:, 2]) - min(coords[:, 2]) + 100
            structure = molecule.get_boxed_structure(a, b, c, no_cross=True, reorder=False)
        else:
            structure = None
        for idx in range(len(molecule)):
            neighbors = strategy.get_nn_info(molecule, idx) if structure is None else strategy.get_nn_info(structure, idx)
            for neighbor in neighbors:
                if not np.array_equal(neighbor['image'], [0, 0, 0]):
                    continue
                if idx > neighbor['site_index']:
                    from_index = neighbor['site_index']
                    to_index = idx
                else:
                    from_index = idx
                    to_index = neighbor['site_index']
                mg.add_edge(from_index=from_index, to_index=to_index, weight=neighbor['weight'], warn_duplicates=False)
        duplicates = []
        for edge in mg.graph.edges:
            if edge[2] != 0:
                duplicates.append(edge)
        for duplicate in duplicates:
            mg.graph.remove_edge(duplicate[0], duplicate[1], key=duplicate[2])
        mg.set_node_attributes()
        return mg

    @classmethod
    @deprecated(from_local_env_strategy, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_local_env_strategy(cls, *args, **kwargs):
        return cls.from_local_env_strategy(*args, **kwargs)

    @property
    def name(self):
        """Name of graph"""
        return self.graph.graph['name']

    @property
    def edge_weight_name(self):
        """Name of the edge weight property of graph"""
        return self.graph.graph['edge_weight_name']

    @property
    def edge_weight_unit(self):
        """Units of the edge weight property of graph"""
        return self.graph.graph['edge_weight_units']

    def add_edge(self, from_index, to_index, weight=None, warn_duplicates=True, edge_properties=None):
        """
        Add edge to graph.

        Since physically a 'bond' (or other connection
        between sites) doesn't have a direction, from_index,
        from_jimage can be swapped with to_index, to_jimage.

        However, images will always be shifted so that
        from_index < to_index and from_jimage becomes (0, 0, 0).

        Args:
            from_index: index of site connecting from
            to_index: index of site connecting to
            weight (float): e.g. bond length
            warn_duplicates (bool): if True, will warn if
                trying to add duplicate edges (duplicate edges will not
                be added in either case)
            edge_properties (dict): any other information to
                store on graph edges, similar to Structure's site_properties
        """
        if to_index < from_index:
            to_index, from_index = (from_index, to_index)
        from_index, to_index = (int(from_index), int(to_index))
        existing_edge_data = self.graph.get_edge_data(from_index, to_index)
        if existing_edge_data and warn_duplicates:
            warnings.warn(f'Trying to add an edge that already exists from site {from_index} to site {to_index}.')
            return
        edge_properties = edge_properties or {}
        if weight:
            self.graph.add_edge(from_index, to_index, weight=weight, **edge_properties)
        else:
            self.graph.add_edge(from_index, to_index, **edge_properties)

    def insert_node(self, idx, species, coords, validate_proximity=False, site_properties=None, edges=None):
        """
        A wrapper around Molecule.insert(), which also incorporates the new
        site into the MoleculeGraph.

        Args:
            idx: Index at which to insert the new site
            species: Species for the new site
            coords: 3x1 array representing coordinates of the new site
            validate_proximity: For Molecule.insert(); if True (default
                False), distance will be checked to ensure that site can be safely
                added.
            site_properties: Site properties for Molecule
            edges: List of dicts representing edges to be added to the
                MoleculeGraph. These edges must include the index of the new site i,
                and all indices used for these edges should reflect the
                MoleculeGraph AFTER the insertion, NOT before. Each dict should at
                least have a "to_index" and "from_index" key, and can also have a
                "weight" and a "properties" key.
        """
        self.molecule.insert(idx, species, coords, validate_proximity=validate_proximity, properties=site_properties)
        mapping = {}
        for j in range(len(self.molecule) - 1):
            if j < idx:
                mapping[j] = j
            else:
                mapping[j] = j + 1
        nx.relabel_nodes(self.graph, mapping, copy=False)
        self.graph.add_node(idx)
        self.set_node_attributes()
        if edges is not None:
            for edge in edges:
                try:
                    self.add_edge(edge['from_index'], edge['to_index'], weight=edge.get('weight'), edge_properties=edge.get('properties'))
                except KeyError:
                    raise RuntimeError('Some edges are invalid.')

    def set_node_attributes(self):
        """
        Replicates molecule site properties (specie, coords, etc.) in the
        MoleculeGraph.
        """
        species = {}
        coords = {}
        properties = {}
        for node in self.graph.nodes():
            species[node] = self.molecule[node].specie.symbol
            coords[node] = self.molecule[node].coords
            properties[node] = self.molecule[node].properties
        nx.set_node_attributes(self.graph, species, 'specie')
        nx.set_node_attributes(self.graph, coords, 'coords')
        nx.set_node_attributes(self.graph, properties, 'properties')

    def alter_edge(self, from_index, to_index, new_weight=None, new_edge_properties=None):
        """
        Alters either the weight or the edge_properties of
        an edge in the MoleculeGraph.

        Args:
            from_index: int
            to_index: int
            new_weight: alter_edge does not require
                that weight be altered. As such, by default, this
                is None. If weight is to be changed, it should be a
                float.
            new_edge_properties: alter_edge does not require
                that edge_properties be altered. As such, by default,
                this is None. If any edge properties are to be changed,
                it should be a dictionary of edge properties to be changed.
        """
        existing_edge = self.graph.get_edge_data(from_index, to_index)
        if not existing_edge:
            raise ValueError(f'Edge between {from_index} and {to_index} cannot be altered; no edge exists between those sites.')
        if new_weight is not None:
            self.graph[from_index][to_index][0]['weight'] = new_weight
        if new_edge_properties is not None:
            for prop in new_edge_properties:
                self.graph[from_index][to_index][0][prop] = new_edge_properties[prop]

    def break_edge(self, from_index, to_index, allow_reverse=False):
        """
        Remove an edge from the MoleculeGraph.

        Args:
            from_index: int
            to_index: int
            allow_reverse: If allow_reverse is True, then break_edge will
                attempt to break both (from_index, to_index) and, failing that,
                will attempt to break (to_index, from_index).
        """
        existing_edge = self.graph.get_edge_data(from_index, to_index)
        existing_reverse = None
        if existing_edge:
            self.graph.remove_edge(from_index, to_index)
        else:
            if allow_reverse:
                existing_reverse = self.graph.get_edge_data(to_index, from_index)
            if existing_reverse:
                self.graph.remove_edge(to_index, from_index)
            else:
                raise ValueError(f'Edge cannot be broken between {from_index} and {to_index}; no edge exists between those sites.')

    def remove_nodes(self, indices: list[int]) -> None:
        """
        A wrapper for Molecule.remove_sites().

        Args:
            indices: indices in the current Molecule (and graph) to be removed.
        """
        self.molecule.remove_sites(indices)
        self.graph.remove_nodes_from(indices)
        mapping = {val: idx for idx, val in enumerate(sorted(self.graph.nodes))}
        nx.relabel_nodes(self.graph, mapping, copy=False)
        self.set_node_attributes()

    def get_disconnected_fragments(self, return_index_map: bool=False):
        """Determine if the MoleculeGraph is connected. If it is not, separate the
        MoleculeGraph into different MoleculeGraphs, where each resulting MoleculeGraph is
        a disconnected subgraph of the original. Currently, this function naively assigns
        the charge of the total molecule to a single submolecule. A later effort will be
        to actually accurately assign charge.

        Args:
            return_index_map (bool): If True, return a dictionary that maps the
                new indices to the original indices. Defaults to False.

        NOTE: This function does not modify the original MoleculeGraph. It creates a copy,
        modifies that, and returns two or more new MoleculeGraph objects.

        Returns:
            list[MoleculeGraph]: Each MoleculeGraph is a disconnected subgraph of the original MoleculeGraph.
        """
        if nx.is_weakly_connected(self.graph):
            return [copy.deepcopy(self)]
        original = copy.deepcopy(self)
        sub_mols = []
        new_to_old_index = []
        for c in nx.weakly_connected_components(original.graph):
            subgraph = original.graph.subgraph(c)
            nodes = sorted(subgraph.nodes)
            new_to_old_index += list(nodes)
            mapping = {val: idx for idx, val in enumerate(nodes)}
            charge = self.molecule.charge if 0 in nodes else 0
            new_graph = nx.relabel_nodes(subgraph, mapping)
            species = nx.get_node_attributes(new_graph, 'specie')
            coords = nx.get_node_attributes(new_graph, 'coords')
            raw_props = nx.get_node_attributes(new_graph, 'properties')
            properties: dict[str, Any] = {}
            for prop_set in raw_props.values():
                for prop in prop_set:
                    if prop in properties:
                        properties[prop].append(prop_set[prop])
                    else:
                        properties[prop] = [prop_set[prop]]
            for k, v in properties.items():
                if len(v) != len(species):
                    del properties[k]
            new_mol = Molecule(species, coords, charge=charge, site_properties=properties)
            graph_data = json_graph.adjacency_data(new_graph)
            sub_mols.append(MoleculeGraph(new_mol, graph_data=graph_data))
        if return_index_map:
            return (sub_mols, dict(enumerate(new_to_old_index)))
        return sub_mols

    def split_molecule_subgraphs(self, bonds, allow_reverse=False, alterations=None):
        """
        Split MoleculeGraph into two or more MoleculeGraphs by
        breaking a set of bonds. This function uses
        MoleculeGraph.break_edge repeatedly to create
        disjoint graphs (two or more separate molecules).
        This function does not only alter the graph
        information, but also changes the underlying
        Molecules.
        If the bonds parameter does not include sufficient
        bonds to separate two molecule fragments, then this
        function will fail.
        Currently, this function naively assigns the charge
        of the total molecule to a single submolecule. A
        later effort will be to actually accurately assign
        charge.
        NOTE: This function does not modify the original
        MoleculeGraph. It creates a copy, modifies that, and
        returns two or more new MoleculeGraph objects.

        Args:
            bonds: list of tuples (from_index, to_index)
                representing bonds to be broken to split the MoleculeGraph.
            alterations: a dict {(from_index, to_index): alt},
                where alt is a dictionary including weight and/or edge
                properties to be changed following the split.
            allow_reverse: If allow_reverse is True, then break_edge will
                attempt to break both (from_index, to_index) and, failing that,
                will attempt to break (to_index, from_index).

        Returns:
            list of MoleculeGraphs.
        """
        self.set_node_attributes()
        original = copy.deepcopy(self)
        for bond in bonds:
            original.break_edge(bond[0], bond[1], allow_reverse=allow_reverse)
        if nx.is_weakly_connected(original.graph):
            raise MolGraphSplitError('Cannot split molecule; MoleculeGraph is still connected.')
        if alterations is not None:
            for u, v in alterations:
                if 'weight' in alterations[u, v]:
                    weight = alterations[u, v].pop('weight')
                    edge_properties = alterations[u, v] if len(alterations[u, v]) != 0 else None
                    original.alter_edge(u, v, new_weight=weight, new_edge_properties=edge_properties)
                else:
                    original.alter_edge(u, v, new_edge_properties=alterations[u, v])
        return original.get_disconnected_fragments()

    def build_unique_fragments(self):
        """
        Find all possible fragment combinations of the MoleculeGraphs (in other
        words, all connected induced subgraphs).
        """
        self.set_node_attributes()
        graph = self.graph.to_undirected()
        frag_dict = {}
        for ii in range(1, len(self.molecule)):
            for combination in combinations(graph.nodes, ii):
                comp = []
                for idx in combination:
                    comp.append(str(self.molecule[idx].specie))
                comp = ''.join(sorted(comp))
                subgraph = nx.subgraph(graph, combination)
                if nx.is_connected(subgraph):
                    key = f'{comp} {len(subgraph.edges())}'
                    if key not in frag_dict:
                        frag_dict[key] = [copy.deepcopy(subgraph)]
                    else:
                        frag_dict[key].append(copy.deepcopy(subgraph))
        unique_frag_dict = {}
        for key, fragments in frag_dict.items():
            unique_frags = []
            for frag in fragments:
                found = False
                for fragment in unique_frags:
                    if _isomorphic(frag, fragment):
                        found = True
                        break
                if not found:
                    unique_frags.append(frag)
            unique_frag_dict[key] = copy.deepcopy(unique_frags)
        unique_mol_graph_dict = {}
        for key, fragments in unique_frag_dict.items():
            unique_mol_graph_list = []
            for fragment in fragments:
                mapping = {edge: idx for idx, edge in enumerate(sorted(fragment.nodes))}
                remapped = nx.relabel_nodes(fragment, mapping)
                species = nx.get_node_attributes(remapped, 'specie')
                coords = nx.get_node_attributes(remapped, 'coords')
                edges = {}
                for from_index, to_index, key in remapped.edges:
                    edge_props = fragment.get_edge_data(from_index, to_index, key=key)
                    edges[from_index, to_index] = edge_props
                unique_mol_graph_list.append(self.from_edges(Molecule(species=species, coords=coords, charge=self.molecule.charge), edges))
            alph_formula = unique_mol_graph_list[0].molecule.composition.alphabetical_formula
            frag_key = f'{alph_formula} E{len(unique_mol_graph_list[0].graph.edges())}'
            unique_mol_graph_dict[frag_key] = copy.deepcopy(unique_mol_graph_list)
        return unique_mol_graph_dict

    def substitute_group(self, index, func_grp, strategy, bond_order=1, graph_dict=None, strategy_params=None):
        """
        Builds off of Molecule.substitute to replace an atom in self.molecule
        with a functional group. This method also amends self.graph to
        incorporate the new functional group.

        NOTE: using a MoleculeGraph will generally produce a different graph
        compared with using a Molecule or str (when not using graph_dict).

        Args:
            index: Index of atom to substitute.
            func_grp: Substituent molecule. There are three options:
                1. Providing an actual molecule as the input. The first atom
                    must be a DummySpecies X, indicating the position of
                    nearest neighbor. The second atom must be the next
                    nearest atom. For example, for a methyl group
                    substitution, func_grp should be X-CH3, where X is the
                    first site and C is the second site. What the code will
                    do is to remove the index site, and connect the nearest
                    neighbor to the C atom in CH3. The X-C bond indicates the
                    directionality to connect the atoms.
                2. A string name. The molecule will be obtained from the
                    relevant template in func_groups.json.
                3. A MoleculeGraph object.
            strategy: Class from pymatgen.analysis.local_env.
            bond_order: A specified bond order to calculate the bond
                length between the attached functional group and the nearest
                neighbor site. Defaults to 1.
            graph_dict: Dictionary representing the bonds of the functional
                group (format: {(u, v): props}, where props is a dictionary of
                properties, including weight. If None, then the algorithm
                will attempt to automatically determine bonds using one of
                a list of strategies defined in pymatgen.analysis.local_env.
            strategy_params: dictionary of keyword arguments for strategy.
                If None, default parameters will be used.
        """

        def map_indices(grp):
            grp_map = {}
            atoms = len(grp) - 1
            offset = len(self.molecule) - atoms
            for idx in range(atoms):
                grp_map[idx] = idx + offset
            return grp_map
        if isinstance(func_grp, MoleculeGraph):
            self.molecule.substitute(index, func_grp.molecule, bond_order=bond_order)
            mapping = map_indices(func_grp.molecule)
            for u, v in list(func_grp.graph.edges()):
                edge_props = func_grp.graph.get_edge_data(u, v)[0]
                weight = edge_props.pop('weight', None)
                self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)
        else:
            if isinstance(func_grp, Molecule):
                func_grp = copy.deepcopy(func_grp)
            else:
                try:
                    func_grp = copy.deepcopy(FunctionalGroups[func_grp])
                except Exception:
                    raise RuntimeError("Can't find functional group in list. Provide explicit coordinate instead")
            self.molecule.substitute(index, func_grp, bond_order=bond_order)
            mapping = map_indices(func_grp)
            func_grp.remove_species('X')
            if graph_dict is not None:
                for u, v in graph_dict:
                    edge_props = graph_dict[u, v]
                    weight = edge_props.pop('weight', None)
                    self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)
            else:
                graph = self.from_local_env_strategy(func_grp, strategy(**strategy_params or {}))
                for u, v in list(graph.graph.edges()):
                    edge_props = graph.graph.get_edge_data(u, v)[0]
                    weight = edge_props.pop('weight', None)
                    if 0 not in list(graph.graph.nodes()):
                        u, v = (u - 1, v - 1)
                    self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)

    def replace_group(self, index, func_grp, strategy, bond_order=1, graph_dict=None, strategy_params=None):
        """
        Builds off of Molecule.substitute and MoleculeGraph.substitute_group
        to replace a functional group in self.molecule with a functional group.
        This method also amends self.graph to incorporate the new functional
        group.

        TODO: Figure out how to replace into a ring structure.

        Args:
            index: Index of atom to substitute.
            func_grp: Substituent molecule. There are three options:
                1. Providing an actual molecule as the input. The first atom
                must be a DummySpecies X, indicating the position of
                nearest neighbor. The second atom must be the next
                nearest atom. For example, for a methyl group
                substitution, func_grp should be X-CH3, where X is the
                first site and C is the second site. What the code will
                do is to remove the index site, and connect the nearest
                neighbor to the C atom in CH3. The X-C bond indicates the
                directionality to connect the atoms.
                2. A string name. The molecule will be obtained from the
                relevant template in func_groups.json.
                3. A MoleculeGraph object.
            strategy: Class from pymatgen.analysis.local_env.
            bond_order: A specified bond order to calculate the bond
                length between the attached functional group and the nearest
                neighbor site. Defaults to 1.
            graph_dict: Dictionary representing the bonds of the functional
                group (format: {(u, v): props}, where props is a dictionary of
                properties, including weight. If None, then the algorithm
                will attempt to automatically determine bonds using one of
                a list of strategies defined in pymatgen.analysis.local_env.
            strategy_params: dictionary of keyword arguments for strategy.
                If None, default parameters will be used.
        """
        self.set_node_attributes()
        neighbors = self.get_connected_sites(index)
        if len(neighbors) == 1:
            self.substitute_group(index, func_grp, strategy, bond_order=bond_order, graph_dict=graph_dict, strategy_params=strategy_params)
        else:
            rings = self.find_rings(including=[index])
            if len(rings) != 0:
                raise RuntimeError('Currently functional group replacement cannot occur at an atom within a ring structure.')
            to_remove = set()
            sizes = {}
            disconnected = self.graph.to_undirected()
            disconnected.remove_node(index)
            for neighbor in neighbors:
                sizes[neighbor[2]] = len(nx.descendants(disconnected, neighbor[2]))
            keep = max(sizes, key=lambda x: sizes[x])
            for idx in sizes:
                if idx != keep:
                    to_remove.add(idx)
            self.remove_nodes(list(to_remove))
            self.substitute_group(index, func_grp, strategy, bond_order=bond_order, graph_dict=graph_dict, strategy_params=strategy_params)

    def find_rings(self, including=None) -> list[list[tuple[int, int]]]:
        """
        Find ring structures in the MoleculeGraph.

        Args:
            including (list[int]): list of site indices. If including is not None, then find_rings
            will only return those rings including the specified sites. By default, this parameter
            is None, and all rings will be returned.

        Returns:
            list[list[tuple[int, int]]]: Each entry will be a ring (cycle, in graph theory terms)
                including the index found in the Molecule. If there is no cycle including an index, the
                value will be an empty list.
        """
        undirected = self.graph.to_undirected()
        directed = undirected.to_directed()
        cycles_nodes = []
        cycles_edges = []
        all_cycles = [sorted(cycle) for cycle in nx.simple_cycles(directed) if len(cycle) > 2]
        unique_sorted = []
        unique_cycles = []
        for cycle in all_cycles:
            if cycle not in unique_sorted:
                unique_sorted.append(cycle)
                unique_cycles.append(cycle)
        if including is None:
            cycles_nodes = unique_cycles
        else:
            for incl in including:
                for cycle in unique_cycles:
                    if incl in cycle and cycle not in cycles_nodes:
                        cycles_nodes.append(cycle)
        for cycle in cycles_nodes:
            edges = []
            for idx, itm in enumerate(cycle, start=-1):
                edges.append((cycle[idx], itm))
            cycles_edges.append(edges)
        return cycles_edges

    def get_connected_sites(self, n):
        """
        Returns a named tuple of neighbors of site n:
        periodic_site, jimage, index, weight.
        Index is the index of the corresponding site
        in the original structure, weight can be
        None if not defined.
        Args:
            n: index of Site in Molecule
            jimage: lattice vector of site

        Returns:
            list of ConnectedSite tuples,
            sorted by closest first.
        """
        connected_sites = set()
        out_edges = list(self.graph.out_edges(n, data=True))
        in_edges = list(self.graph.in_edges(n, data=True))
        for u, v, d in out_edges + in_edges:
            weight = d.get('weight')
            if v == n:
                site = self.molecule[u]
                dist = self.molecule[v].distance(self.molecule[u])
                connected_site = ConnectedSite(site=site, jimage=(0, 0, 0), index=u, weight=weight, dist=dist)
            else:
                site = self.molecule[v]
                dist = self.molecule[u].distance(self.molecule[v])
                connected_site = ConnectedSite(site=site, jimage=(0, 0, 0), index=v, weight=weight, dist=dist)
            connected_sites.add(connected_site)
        connected_sites = list(connected_sites)
        connected_sites.sort(key=lambda x: x.dist)
        return connected_sites

    def get_coordination_of_site(self, n) -> int:
        """
        Returns the number of neighbors of site n.
        In graph terms, simply returns degree
        of node corresponding to site n.

        Args:
            n: index of site

        Returns:
            int: the number of neighbors of site n.
        """
        n_self_loops = sum((1 for n, v in self.graph.edges(n) if n == v))
        return self.graph.degree(n) - n_self_loops

    def draw_graph_to_file(self, filename='graph', diff=None, hide_unconnected_nodes=False, hide_image_edges=True, edge_colors=False, node_labels=False, weight_labels=False, image_labels=False, color_scheme='VESTA', keep_dot=False, algo='fdp'):
        """
        Draws graph using GraphViz.

        The networkx graph object itself can also be drawn
        with networkx's in-built graph drawing methods, but
        note that this might give misleading results for
        multigraphs (edges are super-imposed on each other).

        If visualization is difficult to interpret,
        `hide_image_edges` can help, especially in larger
        graphs.

        Args:
            filename: filename to output, will detect filetype
                from extension (any graphviz filetype supported, such as
                pdf or png)
            diff (StructureGraph): an additional graph to
                compare with, will color edges red that do not exist in diff
                and edges green that are in diff graph but not in the
                reference graph
            hide_unconnected_nodes: if True, hide unconnected nodes
            hide_image_edges: if True, do not draw edges that
                go through periodic boundaries
            edge_colors (bool): if True, use node colors to color edges
            node_labels (bool): if True, label nodes with
                species and site index
            weight_labels (bool): if True, label edges with weights
            image_labels (bool): if True, label edges with
                their periodic images (usually only used for debugging,
                edges to periodic images always appear as dashed lines)
            color_scheme (str): "VESTA" or "JMOL"
            keep_dot (bool): keep GraphViz .dot file for later visualization
            algo: any graphviz algo, "neato" (for simple graphs)
                or "fdp" (for more crowded graphs) usually give good outputs
        """
        if not which(algo):
            raise RuntimeError('StructureGraph graph drawing requires GraphViz binaries to be in the path.')
        g = self.graph.copy()
        g.graph = {'nodesep': 10.0, 'dpi': 300, 'overlap': 'false'}
        for node in g.nodes():
            label = f'{self.molecule[node].specie}({node})' if node_labels else ''
            c = EL_COLORS[color_scheme].get(str(self.molecule[node].specie.symbol), [0, 0, 0])
            fontcolor = '#000000' if 1 - (c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114) / 255 < 0.5 else '#ffffff'
            color = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
            g.add_node(node, fillcolor=color, fontcolor=fontcolor, label=label, fontname='Helvetica-bold', style='filled', shape='circle')
        edges_to_delete = []
        for u, v, k, d in g.edges(keys=True, data=True):
            to_image = d['to_jimage'] if 'to_image' in d else (0, 0, 0)
            d['style'] = 'solid'
            if to_image != (0, 0, 0):
                d['style'] = 'dashed'
                if hide_image_edges:
                    edges_to_delete.append((u, v, k))
            d['arrowhead'] = 'none'
            if image_labels:
                d['headlabel'] = '' if to_image == (0, 0, 0) else f'to {to_image}'
                d['arrowhead'] = 'normal' if d['headlabel'] else 'none'
            color_u = g.nodes[u]['fillcolor']
            color_v = g.nodes[v]['fillcolor']
            d['color_uv'] = f'{color_u};0.5:{color_v};0.5' if edge_colors else '#000000'
            if weight_labels:
                units = g.graph.get('edge_weight_units', '')
                if d.get('weight'):
                    d['label'] = f'{d['weight']:.2f} {units}'
            g.edges[u, v, k].update(d)
        if hide_image_edges:
            for edge_to_delete in edges_to_delete:
                g.remove_edge(*edge_to_delete)
        if hide_unconnected_nodes:
            g = g.subgraph([n for n in g.degree() if g.degree()[n] != 0])
        if diff:
            diff = self.diff(diff, strict=True)
            green_edges = []
            red_edges = []
            for u, v, k, d in g.edges(keys=True, data=True):
                if (u, v, d['to_jimage']) in diff['self']:
                    red_edges.append((u, v, k))
                elif (u, v, d['to_jimage']) in diff['other']:
                    green_edges.append((u, v, k))
            for u, v, k in green_edges:
                g.edges[u, v, k]['color_uv'] = '#00ff00'
            for u, v, k in red_edges:
                g.edges[u, v, k]['color_uv'] = '#ff0000'
        basename, extension = os.path.splitext(filename)
        extension = extension[1:]
        write_dot(g, f'{basename}.dot')
        with open(filename, mode='w') as file:
            args = [algo, '-T', extension, f'{basename}.dot']
            with subprocess.Popen(args, stdout=file, stdin=subprocess.PIPE, close_fds=True) as rs:
                rs.communicate()
                if rs.returncode != 0:
                    raise RuntimeError(f'{algo} exited with return code {rs.returncode}.')
        if not keep_dot:
            os.remove(f'{basename}.dot')

    def as_dict(self):
        """
        As in pymatgen.core.Molecule except
        with using `to_dict_of_dicts` from NetworkX
        to store graph information.
        """
        dct = {'@module': type(self).__module__, '@class': type(self).__name__}
        dct['molecule'] = self.molecule.as_dict()
        dct['graphs'] = json_graph.adjacency_data(self.graph)
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        As in pymatgen.core.Molecule except
        restoring graphs using `from_dict_of_dicts`
        from NetworkX to restore graph information.
        """
        mol = Molecule.from_dict(dct['molecule'])
        return cls(mol, dct['graphs'])

    @classmethod
    def _edges_to_str(cls, g):
        header = 'from    to  to_image    '
        header_line = '----  ----  ------------'
        edge_weight_name = g.graph['edge_weight_name']
        if edge_weight_name:
            print_weights = ['weight']
            edge_label = g.graph['edge_weight_name']
            edge_weight_units = g.graph['edge_weight_units']
            if edge_weight_units:
                edge_label += f' ({edge_weight_units})'
            header += f'  {edge_label}'
            header_line += f'  {'-' * max([18, len(edge_label)])}'
        else:
            print_weights = False
        out = f'{header}\n{header_line}\n'
        edges = list(g.edges(data=True))
        edges.sort(key=itemgetter(0, 1))
        if print_weights:
            for u, v, data in edges:
                out += f'{u:4}  {v:4}  {data.get('to_jimage', (0, 0, 0))!s:12}  {data.get('weight', 0):.3e}\n'
        else:
            for u, v, data in edges:
                out += f'{u:4}  {v:4}  {data.get('to_jimage', (0, 0, 0))!s:12}\n'
        return out

    def __str__(self) -> str:
        out = 'Molecule Graph'
        out += f'\nMolecule: \n{self.molecule}'
        out += f'\nGraph: {self.name}\n'
        out += self._edges_to_str(self.graph)
        return out

    def __repr__(self) -> str:
        out = 'Molecule Graph'
        out += f'\nMolecule: \n{self.molecule!r}'
        out += f'\nGraph: {self.name}\n'
        out += self._edges_to_str(self.graph)
        return out

    def __len__(self) -> int:
        """length of Molecule / number of nodes in graph"""
        return len(self.molecule)

    def sort(self, key: Callable[[Molecule], float] | None=None, reverse: bool=False) -> None:
        """Same as Molecule.sort(). Also remaps nodes in graph.

        Args:
            key (callable, optional): Sort key. Defaults to None.
            reverse (bool, optional): Reverse sort order. Defaults to False.
        """
        old_molecule = self.molecule.copy()
        self.molecule._sites = sorted(self.molecule._sites, key=key, reverse=reverse)
        mapping = {idx: self.molecule.index(site) for idx, site in enumerate(old_molecule)}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        edges_to_remove = []
        edges_to_add = []
        for u, v, keys, data in self.graph.edges(keys=True, data=True):
            if v < u:
                new_v, new_u, new_d = (u, v, data.copy())
                new_d['to_jimage'] = (0, 0, 0)
                edges_to_remove.append((u, v, keys))
                edges_to_add.append((new_u, new_v, new_d))
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
        for u, v, data in edges_to_add:
            self.graph.add_edge(u, v, **data)

    def __copy__(self):
        return type(self).from_dict(self.as_dict())

    def __eq__(self, other: object) -> bool:
        """
        Two MoleculeGraphs are equal if they have equal Molecules,
        and have the same edges between Sites. Edge weights can be
        different and MoleculeGraphs can still be considered equal.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        try:
            mapping = {tuple(site.coords): self.molecule.index(site) for site in other.molecule}
        except ValueError:
            return False
        other_sorted = other.__copy__()
        other_sorted.sort(key=lambda site: mapping[tuple(site.coords)])
        edges = set(self.graph.edges(keys=False))
        edges_other = set(other_sorted.graph.edges(keys=False))
        return edges == edges_other and self.molecule == other_sorted.molecule

    def isomorphic_to(self, other: MoleculeGraph) -> bool:
        """
        Checks if the graphs of two MoleculeGraphs are isomorphic to one
        another. In order to prevent problems with misdirected edges, both
        graphs are converted into undirected nx.Graph objects.

        Args:
            other: MoleculeGraph object to be compared.

        Returns:
            bool
        """
        if len(self.molecule) != len(other.molecule):
            return False
        if self.molecule.composition.alphabetical_formula != other.molecule.composition.alphabetical_formula:
            return False
        if len(self.graph.edges()) != len(other.graph.edges()):
            return False
        return _isomorphic(self.graph, other.graph)

    def diff(self, other, strict=True):
        """
        Compares two MoleculeGraphs. Returns dict with
        keys 'self', 'other', 'both' with edges that are
        present in only one MoleculeGraph ('self' and
        'other'), and edges that are present in both.

        The Jaccard distance is a simple measure of the
        dissimilarity between two MoleculeGraphs (ignoring
        edge weights), and is defined by 1 - (size of the
        intersection / size of the union) of the sets of
        edges. This is returned with key 'dist'.

        Important note: all node indices are in terms
        of the MoleculeGraph this method is called
        from, not the 'other' MoleculeGraph: there
        is no guarantee the node indices will be the
        same if the underlying Molecules are ordered
        differently.

        Args:
            other: MoleculeGraph
            strict: if False, will compare bonds
                from different Molecules, with node indices replaced by Species
                strings, will not count number of occurrences of bonds
        """
        if self.molecule != other.molecule and strict:
            return ValueError('Meaningless to compare MoleculeGraphs if corresponding Molecules are different.')
        if strict:
            mapping = {tuple(site.frac_coords): self.molecule.index(site) for site in other.molecule}
            other_sorted = copy.copy(other)
            other_sorted.sort(key=lambda site: mapping[tuple(site.frac_coords)])
            edges = {(u, v, data.get('to_jimage', (0, 0, 0))) for u, v, data in self.graph.edges(keys=False, data=True)}
            edges_other = {(u, v, data.get('to_jimage', (0, 0, 0))) for u, v, data in other_sorted.graph.edges(keys=False, data=True)}
        else:
            edges = {(str(self.molecule[u].specie), str(self.molecule[v].specie)) for u, v in self.graph.edges(keys=False)}
            edges_other = {(str(other.structure[u].specie), str(other.structure[v].specie)) for u, v in other.graph.edges(keys=False)}
        if len(edges) == 0 and len(edges_other) == 0:
            jaccard_dist = 0
        else:
            jaccard_dist = 1 - len(edges ^ edges_other) / len(edges | edges_other)
        return {'self': edges - edges_other, 'other': edges_other - edges, 'both': edges ^ edges_other, 'dist': jaccard_dist}