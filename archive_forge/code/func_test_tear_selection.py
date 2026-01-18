import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
@unittest.skipIf(not glpk_available, 'GLPK solver not available')
def test_tear_selection(self):
    m = self.simple_recycle_model()
    seq = SequentialDecomposition()
    G = seq.create_graph(m)
    heu_result = seq.select_tear_heuristic(G)
    self.assertEqual(heu_result[1], 1)
    self.assertEqual(heu_result[2], 1)
    all_tsets = []
    for tset in heu_result[0]:
        all_tsets.append(seq.indexes_to_arcs(G, tset))
    for arc in (m.stream_mixer_to_unit, m.stream_unit_to_splitter, m.stream_splitter_to_mixer):
        self.assertIn([arc], all_tsets)
    tset_mip = seq.tear_set_arcs(G, 'mip', solver='glpk')
    self.assertIn(tset_mip, all_tsets)
    tset_heu = seq.tear_set_arcs(G, 'heuristic')
    self.assertIn(tset_heu, all_tsets)