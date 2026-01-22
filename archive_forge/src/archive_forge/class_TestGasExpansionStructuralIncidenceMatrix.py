import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(networkx_available, 'networkx is not available.')
@unittest.skipUnless(scipy_available, 'scipy is not available.')
class TestGasExpansionStructuralIncidenceMatrix(unittest.TestCase):
    """
    This class tests the get_structural_incidence_matrix function
    on the gas expansion model.
    """

    def test_incidence_matrix(self):
        N = 5
        model = make_gas_expansion_model(N)
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(all_vars, all_cons)
        n_var = 4 * (N + 1)
        n_con = 4 * N + 1
        self.assertEqual(imat.shape, (n_con, n_var))
        var_idx_map = ComponentMap(((v, i) for i, v in enumerate(all_vars)))
        con_idx_map = ComponentMap(((c, i) for i, c in enumerate(all_cons)))
        csr_map = ComponentMap()
        csr_map.update(((model.mbal[i], ComponentSet([model.F[i], model.F[i - 1], model.rho[i], model.rho[i - 1]])) for i in model.streams if i != model.streams.first()))
        csr_map.update(((model.ebal[i], ComponentSet([model.F[i], model.F[i - 1], model.rho[i], model.rho[i - 1], model.T[i], model.T[i - 1]])) for i in model.streams if i != model.streams.first()))
        csr_map.update(((model.expansion[i], ComponentSet([model.rho[i], model.rho[i - 1], model.P[i], model.P[i - 1]])) for i in model.streams if i != model.streams.first()))
        csr_map.update(((model.ideal_gas[i], ComponentSet([model.P[i], model.rho[i], model.T[i]])) for i in model.streams))
        i = model.streams.first()
        for i, j, e in zip(imat.row, imat.col, imat.data):
            con = all_cons[i]
            var = all_vars[j]
            self.assertIn(var, csr_map[con])
            csr_map[con].remove(var)
            self.assertEqual(e, 1.0)
        for con in csr_map:
            self.assertEqual(len(csr_map[con]), 0)

    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(all_vars, all_cons)
        n_eqn = len(all_cons)
        matching = maximum_matching(imat)
        values = set(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap(((c, i) for i, c in enumerate(constraints)))
        n_var = len(variables)
        matching = maximum_matching(imat)
        matching = ComponentMap(((c, variables[matching[con_idx_map[c]]]) for c in constraints))
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap(((c, i) for i, c in enumerate(constraints)))
        var_idx_map = ComponentMap(((v, i) for i, v in enumerate(variables)))
        row_block_map, col_block_map = map_coords_to_block_triangular_indices(imat)
        var_block_map = ComponentMap(((v, col_block_map[var_idx_map[v]]) for v in variables))
        con_block_map = ComponentMap(((c, row_block_map[con_idx_map[c]]) for c in constraints))
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)
        self.assertEqual(var_block_map[model.P[0]], 0)
        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)
                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)