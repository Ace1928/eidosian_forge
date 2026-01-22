from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
class TestVariablesWithIndices(unittest.TestCase):
    """Test the DesignVariable class, specify, add_element, add_bounds, update_values."""

    def test_setup(self):
        special = VariablesWithIndices()
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        var_C = 'CA0'
        indices_C = {0: [0]}
        exp1_C = [5]
        special.add_variables(var_C, indices=indices_C, time_index_position=0, values=exp1_C, lower_bounds=1, upper_bounds=5)
        var_T = 'T'
        indices_T = {0: t_control}
        exp1_T = [470, 300, 300, 300, 300, 300, 300, 300, 300]
        special.add_variables(var_T, indices=indices_T, time_index_position=0, values=exp1_T, lower_bounds=300, upper_bounds=700)
        self.assertEqual(special.variable_names, ['CA0[0]', 'T[0]', 'T[0.125]', 'T[0.25]', 'T[0.375]', 'T[0.5]', 'T[0.625]', 'T[0.75]', 'T[0.875]', 'T[1]'])
        self.assertEqual(special.variable_names_value['CA0[0]'], 5)
        self.assertEqual(special.variable_names_value['T[0]'], 470)
        self.assertEqual(special.upper_bounds['CA0[0]'], 5)
        self.assertEqual(special.upper_bounds['T[0]'], 700)
        self.assertEqual(special.lower_bounds['CA0[0]'], 1)
        self.assertEqual(special.lower_bounds['T[0]'], 300)