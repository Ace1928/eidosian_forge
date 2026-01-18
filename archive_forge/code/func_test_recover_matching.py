import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_recover_matching(self):
    N_model = 4
    m = make_gas_expansion_model(N_model)
    variables = list(m.component_data_objects(pyo.Var))
    constraints = list(m.component_data_objects(pyo.Constraint))
    imat = get_structural_incidence_matrix(variables, constraints)
    rdmp, cdmp = dulmage_mendelsohn(imat)
    rmatch = rdmp.underconstrained + rdmp.square + rdmp.overconstrained
    cmatch = cdmp.underconstrained + cdmp.square + cdmp.overconstrained
    matching = list(zip(rmatch, cmatch))
    rmatch = [r for r, c in matching]
    cmatch = [c for r, c in matching]
    self.assertEqual(len(set(rmatch)), len(rmatch))
    self.assertEqual(len(set(cmatch)), len(cmatch))
    entry_set = set(zip(imat.row, imat.col))
    for i, j in matching:
        self.assertIn((i, j), entry_set)