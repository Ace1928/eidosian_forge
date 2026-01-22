import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_openloop import run_cstr_openloop
@unittest.skipIf(not ipopt_available, 'ipopt is not available')
class TestCSTROpenLoop(unittest.TestCase):
    _pred_A_data = [1.0, 1.5, 1.83333, 2.05556, 2.2037, 2.30247, 2.36831, 2.41221, 2.44147, 2.46098, 2.47399, 2.48266, 2.48844, 2.36031, 2.27039, 2.20729, 2.16301, 2.13194, 2.11013, 2.09483, 2.08409, 2.07656, 2.07127, 2.06756, 2.06495, 2.16268, 2.22893, 2.27385, 2.30431, 2.32495, 2.33895, 2.34844, 2.35488, 2.35924, 2.3622, 2.3642, 2.36556, 2.36648, 2.36711, 2.36753, 2.36782]
    _pred_B_data = [0.0, 0.302, 0.61027, 0.90132, 1.1638, 1.39353, 1.59049, 1.75683, 1.89576, 2.01081, 2.10544, 2.18289, 2.246, 2.41517, 2.54001, 2.63284, 2.70242, 2.75502, 2.79516, 2.82605, 2.85007, 2.8689, 2.8838, 2.89569, 2.90526, 2.81484, 2.75455, 2.7145, 2.68802, 2.67062, 2.65927, 2.65195, 2.64728, 2.64436, 2.64258, 2.64153, 2.64096, 2.64067, 2.64057, 2.64058, 2.64064]

    def _get_input_sequence(self):
        input_sequence = mpc.TimeSeriesData({'flow_in[*]': [0.1, 1.0, 0.7, 0.9]}, [0.0, 3.0, 6.0, 10.0])
        return mpc.data.convert.series_to_interval(input_sequence)

    def test_openloop_simulation(self):
        input_sequence = self._get_input_sequence()
        ntfe = 4
        model_horizon = 1.0
        simulation_steps = 10
        m, sim_data = run_cstr_openloop(input_sequence, model_horizon=1.0, ntfe=4, simulation_steps=10)
        sim_time_points = [model_horizon / ntfe * i for i in range(simulation_steps * ntfe + 1)]
        AB_data = sim_data.extract_variables([m.conc[:, 'A'], m.conc[:, 'B']])
        A_cuid = sim_data.get_cuid(m.conc[:, 'A'])
        B_cuid = sim_data.get_cuid(m.conc[:, 'B'])
        pred_data = {A_cuid: self._pred_A_data, B_cuid: self._pred_B_data}
        self.assertStructuredAlmostEqual(pred_data, AB_data.get_data(), delta=0.001)
        self.assertStructuredAlmostEqual(sim_time_points, AB_data.get_time_points(), delta=1e-07)