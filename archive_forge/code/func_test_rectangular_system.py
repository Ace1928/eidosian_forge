import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_rectangular_system(self):
    N_model = 2
    m = make_gas_expansion_model(N_model)
    variables = list(m.component_data_objects(pyo.Var))
    constraints = list(m.component_data_objects(pyo.Constraint))
    imat = get_structural_incidence_matrix(variables, constraints)
    M, N = imat.shape
    self.assertEqual(M, 4 * N_model + 1)
    self.assertEqual(N, 4 * (N_model + 1))
    row_partition, col_partition = dulmage_mendelsohn(imat)
    self.assertEqual(row_partition[0], [])
    self.assertEqual(row_partition[1], [])
    matched_con_set = set(range(len(constraints)))
    self.assertEqual(set(row_partition[2]), matched_con_set)
    self.assertEqual(len(col_partition[0]), 3)
    potentially_unmatched = col_partition[0] + col_partition[1]
    potentially_unmatched_set = set(range(len(variables)))
    self.assertEqual(set(potentially_unmatched), potentially_unmatched_set)