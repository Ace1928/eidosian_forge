import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_square_ill_posed_model(self):
    N = 1
    m = make_gas_expansion_model(N)
    m.P[0].fix()
    m.rho[0].fix()
    m.T[0].fix()
    variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
    constraints = list(m.component_data_objects(pyo.Constraint))
    imat = get_structural_incidence_matrix(variables, constraints)
    var_idx_map = ComponentMap(((v, i) for i, v in enumerate(variables)))
    con_idx_map = ComponentMap(((c, i) for i, c in enumerate(constraints)))
    N, M = imat.shape
    self.assertEqual(N, M)
    row_partition, col_partition = dulmage_mendelsohn(imat)
    unmatched_rows = [con_idx_map[m.ideal_gas[0]]]
    self.assertEqual(row_partition[0], unmatched_rows)
    self.assertEqual(row_partition[1], [])
    matched_con_set = set((con_idx_map[con] for con in constraints if con is not m.ideal_gas[0]))
    self.assertEqual(set(row_partition[2]), matched_con_set)
    potentially_unmatched_set = set(range(len(variables)))
    potentially_unmatched = col_partition[0] + col_partition[1]
    self.assertEqual(set(potentially_unmatched), potentially_unmatched_set)