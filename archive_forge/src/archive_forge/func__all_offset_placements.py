import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def _all_offset_placements(device_graph, offset=(4, 2), min_sidelength=2, max_sidelength=5):
    sidelens = list(itertools.product(range(min_sidelength, max_sidelength + 1), repeat=2))
    topos = [cirq.TiltedSquareLattice(width, height) for width, height in sidelens]
    placements = {topo: topo.nodes_to_gridqubits(offset=offset) for topo in topos}
    placements = {topo: mapping for topo, mapping in placements.items() if cirq.is_valid_placement(device_graph, topo.graph, mapping)}
    return placements