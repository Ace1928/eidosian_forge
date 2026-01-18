import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_rectangular_model(self):
    m = make_dynamic_model()
    m.height[0].fix()
    variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
    constraints = list(m.component_data_objects(pyo.Constraint))
    imat = get_structural_incidence_matrix(variables, constraints)
    M, N = imat.shape
    var_idx_map = ComponentMap(((v, i) for i, v in enumerate(variables)))
    con_idx_map = ComponentMap(((c, i) for i, c in enumerate(constraints)))
    row_partition, col_partition = dulmage_mendelsohn(imat)
    self.assertEqual(row_partition[0], [])
    self.assertEqual(row_partition[1], [])
    self.assertEqual(len(row_partition[3]), 1)
    self.assertEqual(row_partition[3][0], con_idx_map[m.flow_out_eqn[0]])
    self.assertEqual(len(col_partition[3]), 1)
    self.assertEqual(col_partition[3][0], var_idx_map[m.flow_out[0]])
    self.assertEqual(len(row_partition[2]), M - 1)
    row_indices = set([i for i in range(M) if i != con_idx_map[m.flow_out_eqn[0]]])
    self.assertEqual(set(row_partition[2]), row_indices)
    self.assertEqual(len(col_partition[0]), N - M)
    self.assertEqual(len(col_partition[1]), M - 1)
    potentially_unmatched = col_partition[0] + col_partition[1]
    col_indices = set([i for i in range(N) if i != var_idx_map[m.flow_out[0]]])
    self.assertEqual(set(potentially_unmatched), col_indices)