from __future__ import annotations
import copy
import itertools
from collections import defaultdict
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_analyzer import get_max_bond_lengths
from pymatgen.core import Molecule, Species, Structure
from pymatgen.core.lattice import get_integer_index
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def zero_d_graph_to_molecule_graph(bonded_structure, graph):
    """
    Converts a zero-dimensional networkx Graph object into a MoleculeGraph.

    Implements a similar breadth-first search to that in
    calculate_dimensionality_of_site().

    Args:
        bonded_structure (StructureGraph): A structure with bonds, represented
            as a pymatgen structure graph. For example, generated using the
            CrystalNN.get_bonded_structure() method.
        graph (nx.Graph): A networkx `Graph` object for the component of
            interest.

    Returns:
        MoleculeGraph: A MoleculeGraph object of the component.
    """
    seen_indices = []
    sites = []
    start_index = next(iter(graph.nodes()))
    queue = [(start_index, (0, 0, 0), bonded_structure.structure[start_index])]
    while len(queue) > 0:
        comp_i, image_i, site_i = queue.pop(0)
        if comp_i in [x[0] for x in seen_indices]:
            raise ValueError('Graph component is not zero-dimensional')
        seen_indices.append((comp_i, image_i))
        sites.append(site_i)
        for site_j in bonded_structure.get_connected_sites(comp_i, jimage=image_i):
            if (site_j.index, site_j.jimage) not in seen_indices and (site_j.index, site_j.jimage, site_j.site) not in queue:
                queue.append((site_j.index, site_j.jimage, site_j.site))
    indices_ordering = np.argsort([x[0] for x in seen_indices])
    sorted_sites = np.array(sites, dtype=object)[indices_ordering]
    sorted_graph = nx.convert_node_labels_to_integers(graph, ordering='sorted')
    mol = Molecule([s.specie for s in sorted_sites], [s.coords for s in sorted_sites])
    return MoleculeGraph.from_edges(mol, nx.Graph(sorted_graph).edges())