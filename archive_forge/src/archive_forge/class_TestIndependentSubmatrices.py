import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipIf(not networkx_available, 'NetworkX is not available')
@unittest.skipIf(not scipy_available, 'SciPy is not available')
class TestIndependentSubmatrices(unittest.TestCase):

    def test_decomposable_matrix(self):
        """
        The following matrix decomposes into two independent diagonal
        blocks.
        | x         |
        | x x       |
        |     x x   |
        |       x x |
        |         x |
        """
        row = [0, 1, 1, 2, 2, 3, 3, 4]
        col = [0, 0, 1, 2, 3, 3, 4, 4]
        data = [1, 1, 1, 1, 1, 1, 1, 1]
        N = 5
        coo = sp.sparse.coo_matrix((data, (row, col)), shape=(N, N))
        row_blocks, col_blocks = get_independent_submatrices(coo)
        self.assertEqual(len(row_blocks), 2)
        self.assertEqual(len(col_blocks), 2)
        self.assertTrue(set(row_blocks[0]) == {0, 1} or set(row_blocks[1]) == {0, 1})
        self.assertTrue(set(col_blocks[0]) == {0, 1} or set(col_blocks[1]) == {0, 1})
        self.assertTrue(set(row_blocks[0]) == {2, 3, 4} or set(row_blocks[1]) == {2, 3, 4})
        self.assertTrue(set(col_blocks[0]) == {2, 3, 4} or set(col_blocks[1]) == {2, 3, 4})

    def test_decomposable_matrix_permuted(self):
        """
        Same matrix as above, but now permuted into a random order.
        """
        row = [0, 1, 1, 2, 2, 3, 3, 4]
        col = [0, 0, 1, 2, 3, 3, 4, 4]
        data = [1, 1, 1, 1, 1, 1, 1, 1]
        N = 5
        row_perm = list(range(N))
        col_perm = list(range(N))
        random.shuffle(row_perm)
        random.shuffle(col_perm)
        row = [row_perm[i] for i in row]
        col = [col_perm[i] for i in col]
        coo = sp.sparse.coo_matrix((data, (row, col)), shape=(N, N))
        row_blocks, col_blocks = get_independent_submatrices(coo)
        self.assertEqual(len(row_blocks), 2)
        self.assertEqual(len(col_blocks), 2)
        row_set_1 = set((row_perm[0], row_perm[1]))
        row_set_2 = set((row_perm[2], row_perm[3], row_perm[4]))
        col_set_1 = set((col_perm[0], col_perm[1]))
        col_set_2 = set((col_perm[2], col_perm[3], col_perm[4]))
        self.assertTrue(set(row_blocks[0]) == row_set_1 or set(row_blocks[1]) == row_set_1)
        self.assertTrue(set(col_blocks[0]) == col_set_1 or set(col_blocks[1]) == col_set_1)
        self.assertTrue(set(row_blocks[0]) == row_set_2 or set(row_blocks[1]) == row_set_2)
        self.assertTrue(set(col_blocks[0]) == col_set_2 or set(col_blocks[1]) == col_set_2)

    def test_dynamic_model_backward(self):
        m = make_dynamic_model(nfe=5, scheme='BACKWARD')
        m.height[0].fix()
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        variables = list(_generate_variables_in_constraints(constraints))
        con_coord_map = ComponentMap(((con, i) for i, con in enumerate(constraints)))
        var_coord_map = ComponentMap(((var, i) for i, var in enumerate(variables)))
        coo = get_structural_incidence_matrix(variables, constraints)
        row_blocks, col_blocks = get_independent_submatrices(coo)
        rc_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(row_blocks, col_blocks)]
        self.assertEqual(len(rc_blocks), 2)
        t0_var_coords = {var_coord_map[m.flow_out[0]], var_coord_map[m.dhdt[0]], var_coord_map[m.flow_in[0]]}
        t0_con_coords = {con_coord_map[m.flow_out_eqn[0]], con_coord_map[m.diff_eqn[0]]}
        var_blocks = [tuple(sorted(t0_var_coords)), tuple((i for i in range(len(variables)) if i not in t0_var_coords))]
        con_blocks = [tuple(sorted(t0_con_coords)), tuple((i for i in range(len(constraints)) if i not in t0_con_coords))]
        target_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(con_blocks, var_blocks)]
        target_blocks = list(sorted(target_blocks))
        rc_blocks = list(sorted(rc_blocks))
        self.assertEqual(target_blocks, rc_blocks)

    def test_dynamic_model_forward(self):
        m = make_dynamic_model(nfe=5, scheme='FORWARD')
        m.height[0].fix()
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        variables = list(_generate_variables_in_constraints(constraints))
        con_coord_map = ComponentMap(((con, i) for i, con in enumerate(constraints)))
        var_coord_map = ComponentMap(((var, i) for i, var in enumerate(variables)))
        coo = get_structural_incidence_matrix(variables, constraints)
        row_blocks, col_blocks = get_independent_submatrices(coo)
        rc_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(row_blocks, col_blocks)]
        self.assertEqual(len(rc_blocks), 1)