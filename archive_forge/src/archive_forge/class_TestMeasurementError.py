from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import (
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
class TestMeasurementError(unittest.TestCase):

    def test(self):
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        variable_name = 'C'
        indices = {0: ['CA', 'CB', 'CC'], 1: t_control}
        measurements = MeasurementVariables()
        with self.assertRaises(ValueError):
            measurements.add_variables(variable_name, indices=indices, time_index_position=2)