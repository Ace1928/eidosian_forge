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
class StructureGraph(MSONable):
    """
    This is a class for annotating a Structure with bond information, stored in the form
    of a graph. A "bond" does not necessarily have to be a chemical bond, but can store
    any kind of information that connects two Sites.
    """

    def __init__(self, structure: Structure, graph_data: dict | None=None) -> None:
        """
        If constructing this class manually, use the from_empty_graph method or
        from_local_env_strategy method (using an algorithm provided by the local_env
        module, such as O'Keeffe).
        This class that contains connection information: relationships between sites
        represented by a Graph structure, and an associated structure object.

        StructureGraph uses the NetworkX package to store and operate on the graph itself, but
        contains a lot of helper methods to make associating a graph with a given
        crystallographic structure easier.
        Use cases for this include storing bonding information, NMR J-couplings,
        Heisenberg exchange parameters, etc.
        For periodic graphs, class stores information on the graph edges of what lattice
        image the edge belongs to.

        Args:
            structure (Structure): Structure object to be analyzed.
            graph_data (dict): Dictionary containing graph information. Not intended to be
                constructed manually see as_dict method for format.
        """
        if isinstance(structure, StructureGraph):
            graph_data = structure.as_dict()['graphs']
        self.structure = structure
        self.graph = nx.readwrite.json_graph.adjacency_graph(graph_data)
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            for key in ('id', 'key'):
                data.pop(key, None)
            if (to_img := data.get('to_jimage')):
                data['to_jimage'] = tuple(to_img)
            if (from_img := data.get('from_jimage')):
                data['from_jimage'] = tuple(from_img)

    @classmethod
    def from_empty_graph(cls, structure: Structure, name: str='bonds', edge_weight_name: str | None=None, edge_weight_units: str | None=None) -> Self:
        """
        Constructor for an empty StructureGraph, i.e. no edges, containing only nodes corresponding
        to sites in Structure.

        Args:
            structure: A pymatgen Structure object.
            name: Name of the graph, e.g. "bonds".
            edge_weight_name: Name of the edge weights, e.g. "bond_length" or "exchange_constant".
            edge_weight_units: Name of the edge weight units, e.g. "Å" or "eV".

        Returns:
            StructureGraph: an empty graph with no edges, only nodes defined
                that correspond to sites in Structure.
        """
        if edge_weight_name and edge_weight_units is None:
            raise ValueError('Please specify units associated with your edge weights. Can be empty string if arbitrary or dimensionless.')
        graph = nx.MultiDiGraph(edge_weight_name=edge_weight_name, edge_weight_units=edge_weight_units, name=name)
        graph.add_nodes_from(range(len(structure)))
        graph_data = json_graph.adjacency_data(graph)
        return cls(structure, graph_data=graph_data)

    @classmethod
    @deprecated(from_empty_graph, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_empty_graph(cls, *args, **kwargs):
        return cls.from_empty_graph(*args, **kwargs)

    @classmethod
    def from_edges(cls, structure: Structure, edges: dict) -> Self:
        """
        Constructor for MoleculeGraph, using pre-existing or pre-defined edges
        with optional edge parameters.

        Args:
            structure: Structure object
            edges: dict representing the bonds of the functional
                group (format: {(from_index, to_index, from_image, to_image): props},
                where props is a dictionary of properties, including weight.
                Props should be None if no additional properties are to be
                specified.

        Returns:
            sg, a StructureGraph
        """
        struct_graph = cls.from_empty_graph(structure, name='bonds', edge_weight_name='weight', edge_weight_units='')
        for edge, props in edges.items():
            try:
                from_index = edge[0]
                to_index = edge[1]
                from_image = edge[2]
                to_image = edge[3]
            except TypeError:
                raise ValueError('Edges must be given as (from_index, to_index, from_image, to_image) tuples')
            if props is not None:
                weight = props.pop('weight', None)
                if len(props.items()) == 0:
                    props = None
            else:
                weight = None
            nodes = struct_graph.graph.nodes
            if not (from_index in nodes and to_index in nodes):
                raise ValueError('Edges cannot be added if nodes are not present in the graph. Please check your indices.')
            struct_graph.add_edge(from_index, to_index, from_jimage=from_image, to_jimage=to_image, weight=weight, edge_properties=props)
        struct_graph.set_node_attributes()
        return struct_graph

    @classmethod
    @deprecated(from_edges, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_edges(cls, *args, **kwargs):
        return cls.from_edges(*args, **kwargs)

    @classmethod
    def from_local_env_strategy(cls, structure: Structure, strategy: NearNeighbors, weights: bool=False, edge_properties: bool=False) -> Self:
        """
        Constructor for StructureGraph, using a strategy
        from pymatgen.analysis.local_env.

        Args:
            structure: Structure object
            strategy: an instance of a pymatgen.analysis.local_env.NearNeighbors object
            weights(bool): if True, use weights from local_env class (consult relevant class for their meaning)
            edge_properties(bool): if True, edge_properties from neighbors will be used
        """
        if not strategy.structures_allowed:
            raise ValueError('Chosen strategy is not designed for use with structures! Please choose another strategy.')
        struct_graph = cls.from_empty_graph(structure, name='bonds')
        for idx, neighbors in enumerate(strategy.get_all_nn_info(structure)):
            for neighbor in neighbors:
                struct_graph.add_edge(from_index=idx, from_jimage=(0, 0, 0), to_index=neighbor['site_index'], to_jimage=neighbor['image'], weight=neighbor['weight'] if weights else None, edge_properties=neighbor['edge_properties'] if edge_properties else None, warn_duplicates=False)
        return struct_graph

    @classmethod
    @deprecated(from_local_env_strategy, 'Deprecated on 2024-03-29, to be removed on 2025-03-20.')
    def with_local_env_strategy(cls, *args, **kwargs):
        return cls.from_local_env_strategy(*args, **kwargs)

    @property
    def name(self) -> str:
        """Name of graph"""
        return self.graph.graph['name']

    @property
    def edge_weight_name(self) -> str:
        """Name of the edge weight property of graph"""
        return self.graph.graph['edge_weight_name']

    @property
    def edge_weight_unit(self):
        """Units of the edge weight property of graph"""
        return self.graph.graph['edge_weight_units']

    def add_edge(self, from_index: int, to_index: int, from_jimage: tuple[int, int, int]=(0, 0, 0), to_jimage: tuple[int, int, int] | None=None, weight: float | None=None, warn_duplicates: bool=True, edge_properties: dict | None=None) -> None:
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
            from_jimage (tuple of ints): lattice vector of periodic
                image, e.g. (1, 0, 0) for periodic image in +x direction
            to_jimage (tuple of ints): lattice vector of image
            weight (float): e.g. bond length
            warn_duplicates (bool): if True, will warn if
                trying to add duplicate edges (duplicate edges will not
                be added in either case)
            edge_properties (dict): any other information to
                store on graph edges, similar to Structure's site_properties
        """
        if to_index < from_index:
            to_index, from_index = (from_index, to_index)
            to_jimage, from_jimage = (from_jimage, to_jimage)
        if not np.array_equal(from_jimage, (0, 0, 0)):
            shift = from_jimage
            from_jimage = np.subtract(from_jimage, shift)
            to_jimage = np.subtract(to_jimage, shift)
        if to_jimage is None:
            warnings.warn('Please specify to_jimage to be unambiguous, trying to automatically detect.')
            dist, to_jimage = self.structure[from_index].distance_and_image(self.structure[to_index])
            if dist == 0:
                images = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
                dists = []
                for image in images:
                    dists.append(self.structure[from_index].distance_and_image(self.structure[from_index], jimage=image)[0])
                dist = min(dists)
            equiv_sites = self.structure.get_neighbors_in_shell(self.structure[from_index].coords, dist, dist * 0.01, include_index=True)
            for nnsite in equiv_sites:
                to_jimage = np.subtract(nnsite.frac_coords, self.structure[from_index].frac_coords)
                to_jimage = np.round(to_jimage).astype(int)
                self.add_edge(from_index=from_index, from_jimage=(0, 0, 0), to_jimage=to_jimage, to_index=nnsite.index)
            return
        from_jimage, to_jimage = (tuple(map(int, from_jimage)), tuple(map(int, to_jimage)))
        from_index, to_index = (int(from_index), int(to_index))
        if to_index == from_index:
            if to_jimage == (0, 0, 0):
                warnings.warn("Tried to create a bond to itself, this doesn't make sense so was ignored.")
                return
            is_positive = next((idx for idx in to_jimage if idx != 0)) > 0
            if not is_positive:
                to_jimage = tuple((-idx for idx in to_jimage))
        existing_edge_data = self.graph.get_edge_data(from_index, to_index)
        if existing_edge_data:
            for d in existing_edge_data.values():
                if d['to_jimage'] == to_jimage:
                    if warn_duplicates:
                        warnings.warn(f'Trying to add an edge that already exists from site {from_index} to site {to_index} in {to_jimage}.')
                    return
        edge_properties = edge_properties or {}
        if weight:
            self.graph.add_edge(from_index, to_index, to_jimage=to_jimage, weight=weight, **edge_properties)
        else:
            self.graph.add_edge(from_index, to_index, to_jimage=to_jimage, **edge_properties)

    def insert_node(self, idx: int, species: Species, coords: ArrayLike, coords_are_cartesian: bool=False, validate_proximity: bool=False, site_properties: dict | None=None, edges: list | dict | None=None) -> None:
        """
        A wrapper around Molecule.insert(), which also incorporates the new
        site into the MoleculeGraph.

        Args:
            idx: Index at which to insert the new site
            species: Species for the new site
            coords: 3x1 array representing coordinates of the new site
            coords_are_cartesian: Whether coordinates are cartesian.
                Defaults to False.
            validate_proximity: For Molecule.insert(); if True (default
                False), distance will be checked to ensure that
                site can be safely added.
            site_properties: Site properties for Molecule
            edges: List of dicts representing edges to be added to the
            MoleculeGraph. These edges must include the index of the new site i,
            and all indices used for these edges should reflect the
            MoleculeGraph AFTER the insertion, NOT before. Each dict should at
            least have a "to_index" and "from_index" key, and can also have a
            "weight" and a "properties" key.
        """
        self.structure.insert(idx, species, coords, coords_are_cartesian=coords_are_cartesian, validate_proximity=validate_proximity, properties=site_properties)
        mapping = {}
        for j in range(len(self.structure) - 1):
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
                    self.add_edge(edge['from_index'], edge['to_index'], from_jimage=(0, 0, 0), to_jimage=edge['to_jimage'], weight=edge.get('weight'), edge_properties=edge.get('properties'))
                except KeyError:
                    raise RuntimeError('Some edges are invalid.')

    def set_node_attributes(self) -> None:
        """
        Gives each node a "specie" and a "coords" attribute, updated with the
        current species and coordinates.
        """
        species = {}
        coords = {}
        properties = {}
        for node in self.graph.nodes():
            species[node] = self.structure[node].specie.symbol
            coords[node] = self.structure[node].coords
            properties[node] = self.structure[node].properties
        nx.set_node_attributes(self.graph, species, 'specie')
        nx.set_node_attributes(self.graph, coords, 'coords')
        nx.set_node_attributes(self.graph, properties, 'properties')

    def alter_edge(self, from_index: int, to_index: int, to_jimage: tuple | None=None, new_weight: float | None=None, new_edge_properties: dict | None=None):
        """
        Alters either the weight or the edge_properties of
        an edge in the StructureGraph.

        Args:
            from_index: int
            to_index: int
            to_jimage: tuple
            new_weight: alter_edge does not require
                that weight be altered. As such, by default, this
                is None. If weight is to be changed, it should be a
                float.
            new_edge_properties: alter_edge does not require
                that edge_properties be altered. As such, by default,
                this is None. If any edge properties are to be changed,
                it should be a dictionary of edge properties to be changed.
        """
        existing_edges = self.graph.get_edge_data(from_index, to_index)
        if not existing_edges:
            raise ValueError(f'Edge between {from_index} and {to_index} cannot be altered; no edge exists between those sites.')
        if to_jimage is None:
            edge_index = 0
        else:
            for idx, properties in existing_edges.items():
                if properties['to_jimage'] == to_jimage:
                    edge_index = idx
        if new_weight is not None:
            self.graph[from_index][to_index][edge_index]['weight'] = new_weight
        if new_edge_properties is not None:
            for prop in list(new_edge_properties):
                self.graph[from_index][to_index][edge_index][prop] = new_edge_properties[prop]

    def break_edge(self, from_index: int, to_index: int, to_jimage: tuple | None=None, allow_reverse: bool=False) -> None:
        """
        Remove an edge from the StructureGraph. If no image is given, this method will fail.

        Args:
            from_index: int
            to_index: int
            to_jimage: tuple
            allow_reverse: If allow_reverse is True, then break_edge will
                attempt to break both (from_index, to_index) and, failing that,
                will attempt to break (to_index, from_index).
        """
        existing_edges = self.graph.get_edge_data(from_index, to_index)
        existing_reverse = None
        if to_jimage is None:
            raise ValueError('Image must be supplied, to avoid ambiguity.')
        if existing_edges:
            for idx, props in existing_edges.items():
                if props['to_jimage'] == to_jimage:
                    edge_index = idx
            self.graph.remove_edge(from_index, to_index, edge_index)
        else:
            if allow_reverse:
                existing_reverse = self.graph.get_edge_data(to_index, from_index)
            if existing_reverse:
                for idx, props in existing_reverse.items():
                    if props['to_jimage'] == to_jimage:
                        edge_index = idx
                self.graph.remove_edge(to_index, from_index, edge_index)
            else:
                raise ValueError(f'Edge cannot be broken between {from_index} and {to_index}; no edge exists between those sites.')

    def remove_nodes(self, indices: Sequence[int | None]) -> None:
        """
        A wrapper for Molecule.remove_sites().

        Args:
            indices: list of indices in the current Molecule (and graph) to
                be removed.
        """
        self.structure.remove_sites(indices)
        self.graph.remove_nodes_from(indices)
        mapping = {val: idx for idx, val in enumerate(sorted(self.graph.nodes))}
        nx.relabel_nodes(self.graph, mapping, copy=False)
        self.set_node_attributes()

    def substitute_group(self, index: int, func_grp: Molecule | str, strategy: Any, bond_order: int=1, graph_dict: dict | None=None, strategy_params: dict | None=None):
        """
        Builds off of Structure.substitute to replace an atom in self.structure
        with a functional group. This method also amends self.graph to
        incorporate the new functional group.

        NOTE: Care must be taken to ensure that the functional group that is
        substituted will not place atoms to close to each other, or violate the
        dimensions of the Lattice.

        Args:
            index: Index of atom to substitute.
            func_grp: Substituent molecule. There are two options:
                1. Providing an actual Molecule as the input. The first atom
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

        def map_indices(grp: Molecule) -> dict[int, int]:
            grp_map = {}
            atoms = len(grp) - 1
            offset = len(self.structure) - atoms
            for idx in range(atoms):
                grp_map[idx] = idx + offset
            return grp_map
        if isinstance(func_grp, Molecule):
            func_grp = copy.deepcopy(func_grp)
        else:
            try:
                func_grp = copy.deepcopy(FunctionalGroups[func_grp])
            except Exception:
                raise RuntimeError("Can't find functional group in list. Provide explicit coordinate instead")
        self.structure.substitute(index, func_grp, bond_order=bond_order)
        mapping = map_indices(func_grp)
        func_grp.remove_species('X')
        if graph_dict is not None:
            for u, v in graph_dict:
                edge_props = graph_dict[u, v]
                to_jimage = edge_props.get('to_jimage', (0, 0, 0))
                weight = edge_props.pop('weight', None)
                self.add_edge(mapping[u], mapping[v], to_jimage=to_jimage, weight=weight, edge_properties=edge_props)
        else:
            if strategy_params is None:
                strategy_params = {}
            strat = strategy(**strategy_params)
            for site in mapping.values():
                neighbors = strat.get_nn_info(self.structure, site)
                for neighbor in neighbors:
                    self.add_edge(from_index=site, from_jimage=(0, 0, 0), to_index=neighbor['site_index'], to_jimage=neighbor['image'], weight=neighbor['weight'], warn_duplicates=False)

    def get_connected_sites(self, n: int, jimage: tuple[int, int, int]=(0, 0, 0)) -> list[ConnectedSite]:
        """
        Returns a named tuple of neighbors of site n:
        periodic_site, jimage, index, weight.
        Index is the index of the corresponding site
        in the original structure, weight can be
        None if not defined.

        Args:
            n: index of Site in Structure
            jimage: lattice vector of site

        Returns:
            list of ConnectedSite tuples,
            sorted by closest first.
        """
        connected_sites = set()
        connected_site_images = set()
        out_edges = [(u, v, d, 'out') for u, v, d in self.graph.out_edges(n, data=True)]
        in_edges = [(u, v, d, 'in') for u, v, d in self.graph.in_edges(n, data=True)]
        for u, v, data, dir in out_edges + in_edges:
            to_jimage = data['to_jimage']
            if dir == 'in':
                u, v = (v, u)
                to_jimage = np.multiply(-1, to_jimage)
            to_jimage = tuple(map(int, np.add(to_jimage, jimage)))
            site_d = self.structure[v].as_dict()
            site_d['abc'] = np.add(site_d['abc'], to_jimage).tolist()
            site = PeriodicSite.from_dict(site_d)
            relative_jimage = np.subtract(to_jimage, jimage)
            u_site = cast(PeriodicSite, self.structure[u])
            dist = u_site.distance(self.structure[v], jimage=relative_jimage)
            weight = data.get('weight')
            if (v, to_jimage) not in connected_site_images:
                connected_site = ConnectedSite(site=site, jimage=to_jimage, index=v, weight=weight, dist=dist)
                connected_sites.add(connected_site)
                connected_site_images.add((v, to_jimage))
        _connected_sites = list(connected_sites)
        _connected_sites.sort(key=lambda x: x.dist)
        return _connected_sites

    def get_coordination_of_site(self, n: int) -> int:
        """
        Returns the number of neighbors of site n. In graph terms,
        simply returns degree of node corresponding to site n.

        Args:
            n: index of site

        Returns:
            int: number of neighbors of site n.
        """
        n_self_loops = sum((1 for n, v in self.graph.edges(n) if n == v))
        return self.graph.degree(n) - n_self_loops

    def draw_graph_to_file(self, filename: str='graph', diff: StructureGraph=None, hide_unconnected_nodes: bool=False, hide_image_edges: bool=True, edge_colors: bool=False, node_labels: bool=False, weight_labels: bool=False, image_labels: bool=False, color_scheme: str='VESTA', keep_dot: bool=False, algo: str='fdp'):
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
            node_labels (bool): if True, label nodes with species and site index
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
            label = f'{self.structure[node].specie}({node})' if node_labels else ''
            c = EL_COLORS[color_scheme].get(str(self.structure[node].specie.symbol), [0, 0, 0])
            fontcolor = '#000000' if 1 - (c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114) / 255 < 0.5 else '#ffffff'
            color = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
            g.add_node(node, fillcolor=color, fontcolor=fontcolor, label=label, fontname='Helvetica-bold', style='filled', shape='circle')
        edges_to_delete = []
        for u, v, k, d in g.edges(keys=True, data=True):
            to_image = d['to_jimage']
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
            _diff = self.diff(diff, strict=True)
            green_edges = []
            red_edges = []
            for u, v, k, d in g.edges(keys=True, data=True):
                if (u, v, d['to_jimage']) in _diff['self']:
                    red_edges.append((u, v, k))
                elif (u, v, d['to_jimage']) in _diff['other']:
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

    @property
    def types_and_weights_of_connections(self) -> dict:
        """
        Extract a dictionary summarizing the types and weights
        of edges in the graph.

        Returns:
            A dictionary with keys specifying the
            species involved in a connection in alphabetical order
            (e.g. string 'Fe-O') and values which are a list of
            weights for those connections (e.g. bond lengths).
        """

        def get_label(u, v):
            u_label = self.structure[u].species_string
            v_label = self.structure[v].species_string
            return '-'.join(sorted((u_label, v_label)))
        types = defaultdict(list)
        for u, v, d in self.graph.edges(data=True):
            label = get_label(u, v)
            types[label].append(d['weight'])
        return dict(types)

    @property
    def weight_statistics(self) -> dict:
        """
        Extract a statistical summary of edge weights present in
        the graph.

        Returns:
            A dict with an 'all_weights' list, 'minimum',
            'maximum', 'median', 'mean', 'std_dev'
        """
        all_weights = [d.get('weight') for u, v, d in self.graph.edges(data=True)]
        stats = describe(all_weights, nan_policy='omit')
        return {'all_weights': all_weights, 'min': stats.minmax[0], 'max': stats.minmax[1], 'mean': stats.mean, 'variance': stats.variance}

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

    def as_dict(self) -> dict:
        """
        As in pymatgen.core.Structure except
        with using `to_dict_of_dicts` from NetworkX
        to store graph information.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self.structure.as_dict(), 'graphs': json_graph.adjacency_data(self.graph)}

    @classmethod
    def from_dict(cls, dct) -> Self:
        """As in pymatgen.core.Structure except restoring graphs using from_dict_of_dicts
        from NetworkX to restore graph information.
        """
        struct = Structure.from_dict(dct['structure'])
        return cls(struct, dct['graphs'])

    def __mul__(self, scaling_matrix):
        """
        Replicates the graph, creating a supercell,
        intelligently joining together
        edges that lie on periodic boundaries.
        In principle, any operations on the expanded
        graph could also be done on the original
        graph, but a larger graph can be easier to
        visualize and reason about.

        Args:
            scaling_matrix: same as Structure.__mul__
        """
        scale_matrix = np.array(scaling_matrix, int)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), int)
        else:
            raise NotImplementedError('Not tested with 3x3 scaling matrices yet.')
        new_lattice = Lattice(np.dot(scale_matrix, self.structure.lattice.matrix))
        frac_lattice = lattice_points_in_supercell(scale_matrix)
        cart_lattice = new_lattice.get_cartesian_coords(frac_lattice)
        new_sites = []
        new_graphs = []
        for v in cart_lattice:
            mapping = {n: n + len(new_sites) for n in range(len(self.structure))}
            for site in self.structure:
                site = PeriodicSite(site.species, site.coords + v, new_lattice, properties=site.properties, coords_are_cartesian=True, to_unit_cell=False)
                new_sites.append(site)
            new_graphs.append(nx.relabel_nodes(self.graph, mapping, copy=True))
        new_structure = Structure.from_sites(new_sites)
        new_g = nx.MultiDiGraph()
        for new_graph in new_graphs:
            new_g = nx.union(new_g, new_graph)
        edges_to_remove = []
        edges_to_add = []
        edges_inside_supercell = [{u, v} for u, v, d in new_g.edges(data=True) if d['to_jimage'] == (0, 0, 0)]
        new_periodic_images = []
        orig_lattice = self.structure.lattice
        kd_tree = KDTree(new_structure.cart_coords)
        tol = 0.05
        for u, v, k, data in new_g.edges(keys=True, data=True):
            to_jimage = data['to_jimage']
            if to_jimage != (0, 0, 0):
                n_u = u % len(self.structure)
                n_v = v % len(self.structure)
                v_image_frac = np.add(self.structure[n_v].frac_coords, to_jimage)
                u_frac = self.structure[n_u].frac_coords
                v_image_cart = orig_lattice.get_cartesian_coords(v_image_frac)
                u_cart = orig_lattice.get_cartesian_coords(u_frac)
                v_rel = np.subtract(v_image_cart, u_cart)
                v_expect = new_structure[u].coords + v_rel
                v_present = kd_tree.query(v_expect)
                v_present = v_present[1] if v_present[0] <= tol else None
                if v_present is not None:
                    new_u = u
                    new_v = v_present
                    new_data = data.copy()
                    new_data['to_jimage'] = (0, 0, 0)
                    edges_to_remove.append((u, v, k))
                    if {new_u, new_v} not in edges_inside_supercell:
                        if new_v < new_u:
                            new_u, new_v = (new_v, new_u)
                        edges_inside_supercell.append({new_u, new_v})
                        edges_to_add.append((new_u, new_v, new_data))
                else:
                    v_expec_frac = new_structure.lattice.get_fractional_coords(v_expect)
                    v_expec_image = np.around(v_expec_frac, decimals=3)
                    v_expec_image = v_expec_image - v_expec_image % 1
                    v_expec_frac = np.subtract(v_expec_frac, v_expec_image)
                    v_expect = new_structure.lattice.get_cartesian_coords(v_expec_frac)
                    v_present = kd_tree.query(v_expect)
                    v_present = v_present[1] if v_present[0] <= tol else None
                    if v_present is not None:
                        new_u = u
                        new_v = v_present
                        new_data = data.copy()
                        new_to_jimage = tuple(map(int, v_expec_image))
                        if new_v < new_u:
                            new_u, new_v = (new_v, new_u)
                            new_to_jimage = tuple(np.multiply(-1, data['to_jimage']).astype(int))
                        new_data['to_jimage'] = new_to_jimage
                        edges_to_remove.append((u, v, k))
                        if (new_u, new_v, new_to_jimage) not in new_periodic_images:
                            edges_to_add.append((new_u, new_v, new_data))
                            new_periodic_images.append((new_u, new_v, new_to_jimage))
        logger.debug(f'Removing {len(edges_to_remove)} edges, adding {len(edges_to_add)} new edges.')
        for edge in edges_to_remove:
            new_g.remove_edge(*edge)
        for u, v, data in edges_to_add:
            new_g.add_edge(u, v, **data)
        data = {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': new_structure.as_dict(), 'graphs': json_graph.adjacency_data(new_g)}
        return type(self).from_dict(data)

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def _edges_to_str(cls, g) -> str:
        header = 'from    to  to_image    '
        header_line = '----  ----  ------------'
        edge_weight_name = g.graph['edge_weight_name']
        if edge_weight_name:
            print_weights = True
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

    def __str__(self):
        out = 'Structure Graph'
        out += f'\nStructure: \n{self.structure}'
        out += f'\nGraph: {self.name}\n'
        out += self._edges_to_str(self.graph)
        return out

    def __repr__(self):
        out = 'Structure Graph'
        out += f'\nStructure: \n{self.structure!r}'
        out += f'\nGraph: {self.name}\n'
        out += self._edges_to_str(self.graph)
        return out

    def __len__(self):
        """length of Structure / number of nodes in graph"""
        return len(self.structure)

    def sort(self, key=None, reverse: bool=False) -> None:
        """Same as Structure.sort(). Also remaps nodes in graph.

        Args:
            key: key to sort by
            reverse: reverse sort order
        """
        old_structure = self.structure.copy()
        self.structure._sites = sorted(self.structure._sites, key=key, reverse=reverse)
        mapping = {idx: self.structure.index(site) for idx, site in enumerate(old_structure)}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        edges_to_remove = []
        edges_to_add = []
        for u, v, keys, data in self.graph.edges(keys=True, data=True):
            if v < u:
                new_v, new_u, new_d = (u, v, data.copy())
                new_d['to_jimage'] = tuple(np.multiply(-1, data['to_jimage']).astype(int))
                edges_to_remove.append((u, v, keys))
                edges_to_add.append((new_u, new_v, new_d))
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
        for u, v, d in edges_to_add:
            self.graph.add_edge(u, v, **d)

    def __copy__(self):
        return type(self).from_dict(self.as_dict())

    def __eq__(self, other: object) -> bool:
        """
        Two StructureGraphs are equal if they have equal Structures,
        and have the same edges between Sites. Edge weights can be
        different and StructureGraphs can still be considered equal.

        Args:
            other: StructureGraph
        """
        if not isinstance(other, StructureGraph):
            return NotImplemented
        mapping = {tuple(site.frac_coords): self.structure.index(site) for site in other.structure}
        other_sorted = other.__copy__()
        other_sorted.sort(key=lambda site: mapping[tuple(site.frac_coords)])
        edges = {(u, v, data['to_jimage']) for u, v, data in self.graph.edges(keys=False, data=True)}
        edges_other = {(u, v, data['to_jimage']) for u, v, data in other_sorted.graph.edges(keys=False, data=True)}
        return edges == edges_other and self.structure == other_sorted.structure

    def diff(self, other: StructureGraph, strict: bool=True) -> dict:
        """
        Compares two StructureGraphs. Returns dict with
        keys 'self', 'other', 'both' with edges that are
        present in only one StructureGraph ('self' and
        'other'), and edges that are present in both.

        The Jaccard distance is a simple measure of the
        dissimilarity between two StructureGraphs (ignoring
        edge weights), and is defined by 1 - (size of the
        intersection / size of the union) of the sets of
        edges. This is returned with key 'dist'.

        Important note: all node indices are in terms
        of the StructureGraph this method is called
        from, not the 'other' StructureGraph: there
        is no guarantee the node indices will be the
        same if the underlying Structures are ordered
        differently.

        Args:
            other: StructureGraph
            strict: if False, will compare bonds
                from different Structures, with node indices
                replaced by Species strings, will not count
                number of occurrences of bonds
        """
        if self.structure != other.structure and strict:
            raise ValueError('Meaningless to compare StructureGraphs if corresponding Structures are different.')
        if strict:
            mapping = {tuple(site.frac_coords): self.structure.index(site) for site in other.structure}
            other_sorted = copy.copy(other)
            other_sorted.sort(key=lambda site: mapping[tuple(site.frac_coords)])
            edges: set[tuple] = {(u, v, data['to_jimage']) for u, v, data in self.graph.edges(keys=False, data=True)}
            edges_other: set[tuple] = {(u, v, data['to_jimage']) for u, v, data in other_sorted.graph.edges(keys=False, data=True)}
        else:
            edges = {(str(self.structure[u].specie), str(self.structure[v].specie)) for u, v in self.graph.edges(keys=False)}
            edges_other = {(str(other.structure[u].specie), str(other.structure[v].specie)) for u, v in other.graph.edges(keys=False)}
        if len(edges) == 0 and len(edges_other) == 0:
            jaccard_dist = 0.0
        else:
            jaccard_dist = 1.0 - len(edges & edges_other) / len(edges | edges_other)
        return {'self': edges - edges_other, 'other': edges_other - edges, 'both': edges.intersection(edges_other), 'dist': jaccard_dist}

    def get_subgraphs_as_molecules(self, use_weights: bool=False) -> list[Molecule]:
        """
        Retrieve subgraphs as molecules, useful for extracting
        molecules from periodic crystals.

        Will only return unique molecules, not any duplicates
        present in the crystal (a duplicate defined as an
        isomorphic subgraph).

        Args:
            use_weights (bool): If True, only treat subgraphs
                as isomorphic if edges have the same weights. Typically,
                this means molecules will need to have the same bond
                lengths to be defined as duplicates, otherwise bond
                lengths can differ. This is a fairly robust approach,
                but will treat e.g. enantiomers as being duplicates.

        Returns:
            list of unique Molecules in Structure
        """
        if getattr(self, '_supercell_sg', None) is None:
            self._supercell_sg = supercell_sg = self * (3, 3, 3)
        supercell_sg.graph = nx.Graph(supercell_sg.graph)
        all_subgraphs = [supercell_sg.graph.subgraph(c) for c in nx.connected_components(supercell_sg.graph)]
        molecule_subgraphs = []
        for subgraph in all_subgraphs:
            intersects_boundary = any((d['to_jimage'] != (0, 0, 0) for u, v, d in subgraph.edges(data=True)))
            if not intersects_boundary:
                molecule_subgraphs.append(nx.MultiDiGraph(subgraph))
        for subgraph in molecule_subgraphs:
            for n in subgraph:
                subgraph.add_node(n, specie=str(supercell_sg.structure[n].specie))

        def node_match(n1, n2):
            return n1['specie'] == n2['specie']

        def edge_match(e1, e2):
            if use_weights:
                return e1['weight'] == e2['weight']
            return True
        unique_subgraphs: list = []
        for subgraph in molecule_subgraphs:
            already_present = [nx.is_isomorphic(subgraph, g, node_match=node_match, edge_match=edge_match) for g in unique_subgraphs]
            if not any(already_present):
                unique_subgraphs.append(subgraph)
        molecules = []
        for subgraph in unique_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            molecule = Molecule(species, coords)
            molecule = molecule.get_centered_molecule()
            molecules.append(molecule)
        return molecules