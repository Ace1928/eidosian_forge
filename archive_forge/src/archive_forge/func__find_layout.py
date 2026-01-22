import itertools
from qiskit.transpiler import CouplingMap, Target, AnalysisPass, TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit._accelerate.error_map import ErrorMap
def _find_layout(self, dag, edges):
    """Checks if there is a layout for a given set of edges."""
    cm = CouplingMap(edges)
    pass_ = VF2Layout(cm, seed=0, max_trials=1, call_limit=self.call_limit_vf2)
    pass_.run(dag)
    return pass_.property_set.get('layout', None)