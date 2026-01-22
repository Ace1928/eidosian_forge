import itertools
from qiskit.transpiler import CouplingMap, Target, AnalysisPass, TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit._accelerate.error_map import ErrorMap
def _add_extra_edges(self, distance):
    """Augments the coupling map with extra edges that connect nodes ``distance``
        apart in the original graph. The extra edges are assigned errors allowing VF2
        to prioritize real edges over extra edges.
        """
    nq = len(self.coupling_map.graph)
    augmented_coupling_map = CouplingMap()
    augmented_coupling_map.graph = self.coupling_map.graph.copy()
    augmented_error_map = ErrorMap(nq)
    for x, y in itertools.combinations(self.coupling_map.graph.node_indices(), 2):
        d = self.coupling_map.distance(x, y)
        if 1 < d <= distance:
            error_rate = 1 - (1 - self.error_rate) ** d
            augmented_coupling_map.add_edge(x, y)
            augmented_error_map.add_error((x, y), error_rate)
            augmented_coupling_map.add_edge(y, x)
            augmented_error_map.add_error((y, x), error_rate)
    return (augmented_coupling_map, augmented_error_map)