import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise

    Helper - generates all k-edge-components using the aux graph.  Checks the
    both local and subgraph edge connectivity of each cc. Also checks that
    alternate methods of computing the k-edge-ccs generate the same result.
    