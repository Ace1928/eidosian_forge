from collections.abc import Sequence as SequenceType
from itertools import compress
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import numpy as np
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.operation import Operation
Converts a ``MultiDiGraph`` into the
    `hMETIS hypergraph input format <http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf>`__
    conforming to KaHyPar's calling signature.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Defaults to 0 which leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together.
        edge_weights (Sequence[int]): Weights for regular edges in the graph. Defaults to ``None``,
            which leads to unit-weighted edges.

    Returns:
        Tuple[List,List,List]: The 3 lists representing an (optionally weighted) hypergraph:
        - Flattened list of adjacent node indices.
        - List of starting indices for edges in the above adjacent-nodes-list.
        - Optional list of edge weights. ``None`` if ``hyperwire_weight`` is equal to 0.
    