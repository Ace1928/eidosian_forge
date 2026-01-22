from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesign_DAE(unittest.TestCase):

    def setUp(self):

        def ABC_model(data):
            ca_meas = data['ca']
            cb_meas = data['cb']
            cc_meas = data['cc']
            if isinstance(data, pd.DataFrame):
                meas_t = data.index
            else:
                meas_t = list(ca_meas.keys())
            ca0 = 1.0
            cb0 = 0.0
            cc0 = 0.0
            m = pyo.ConcreteModel()
            m.k1 = pyo.Var(initialize=0.5, bounds=(0.0001, 10))
            m.k2 = pyo.Var(initialize=3.0, bounds=(0.0001, 10))
            m.time = dae.ContinuousSet(bounds=(0.0, 5.0), initialize=meas_t)
            m.ca = pyo.Var(m.time, initialize=ca0, bounds=(-0.001, ca0 + 0.001))
            m.cb = pyo.Var(m.time, initialize=cb0, bounds=(-0.001, ca0 + 0.001))
            m.cc = pyo.Var(m.time, initialize=cc0, bounds=(-0.001, ca0 + 0.001))
            m.dca = dae.DerivativeVar(m.ca, wrt=m.time)
            m.dcb = dae.DerivativeVar(m.cb, wrt=m.time)
            m.dcc = dae.DerivativeVar(m.cc, wrt=m.time)

            def _dcarate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dca[t] == -m.k1 * m.ca[t]
            m.dcarate = pyo.Constraint(m.time, rule=_dcarate)

            def _dcbrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcb[t] == m.k1 * m.ca[t] - m.k2 * m.cb[t]
            m.dcbrate = pyo.Constraint(m.time, rule=_dcbrate)

            def _dccrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcc[t] == m.k2 * m.cb[t]
            m.dccrate = pyo.Constraint(m.time, rule=_dccrate)

            def ComputeFirstStageCost_rule(m):
                return 0
            m.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

            def ComputeSecondStageCost_rule(m):
                return sum(((m.ca[t] - ca_meas[t]) ** 2 + (m.cb[t] - cb_meas[t]) ** 2 + (m.cc[t] - cc_meas[t]) ** 2 for t in meas_t))
            m.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

            def total_cost_rule(model):
                return model.FirstStageCost + model.SecondStageCost
            m.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
            disc = pyo.TransformationFactory('dae.collocation')
            disc.apply_to(m, nfe=20, ncp=2)
            return m
        data = [[0.0, 0.957, -0.031, -0.015], [0.263, 0.557, 0.33, 0.044], [0.526, 0.342, 0.512, 0.156], [0.789, 0.224, 0.499, 0.31], [1.053, 0.123, 0.428, 0.454], [1.316, 0.079, 0.396, 0.556], [1.579, 0.035, 0.303, 0.651], [1.842, 0.029, 0.287, 0.658], [2.105, 0.025, 0.221, 0.75], [2.368, 0.017, 0.148, 0.854], [2.632, -0.002, 0.182, 0.845], [2.895, 0.009, 0.116, 0.893], [3.158, -0.023, 0.079, 0.942], [3.421, 0.006, 0.078, 0.899], [3.684, 0.016, 0.059, 0.942], [3.947, 0.014, 0.036, 0.991], [4.211, -0.009, 0.014, 0.988], [4.474, -0.03, 0.036, 0.941], [4.737, 0.004, 0.036, 0.971], [5.0, -0.024, 0.028, 0.985]]
        data = pd.DataFrame(data, columns=['t', 'ca', 'cb', 'cc'])
        data_df = data.set_index('t')
        data_dict = {'ca': {k: v for k, v in zip(data.t, data.ca)}, 'cb': {k: v for k, v in zip(data.t, data.cb)}, 'cc': {k: v for k, v in zip(data.t, data.cc)}}
        theta_names = ['k1', 'k2']
        self.pest_df = parmest.Estimator(ABC_model, [data_df], theta_names)
        self.pest_dict = parmest.Estimator(ABC_model, [data_dict], theta_names)
        self.pest_df_multiple = parmest.Estimator(ABC_model, [data_df, data_df], theta_names)
        self.pest_dict_multiple = parmest.Estimator(ABC_model, [data_dict, data_dict], theta_names)
        self.m_df = ABC_model(data_df)
        self.m_dict = ABC_model(data_dict)

    def test_dataformats(self):
        obj1, theta1 = self.pest_df.theta_est()
        obj2, theta2 = self.pest_dict.theta_est()
        self.assertAlmostEqual(obj1, obj2, places=6)
        self.assertAlmostEqual(theta1['k1'], theta2['k1'], places=6)
        self.assertAlmostEqual(theta1['k2'], theta2['k2'], places=6)

    def test_return_continuous_set(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df.theta_est(return_values=['time'])
        obj2, theta2, return_vals2 = self.pest_dict.theta_est(return_values=['time'])
        self.assertAlmostEqual(return_vals1['time'].loc[0][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2['time'].loc[0][18], 2.368, places=3)

    def test_return_continuous_set_multiple_datasets(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df_multiple.theta_est(return_values=['time'])
        obj2, theta2, return_vals2 = self.pest_dict_multiple.theta_est(return_values=['time'])
        self.assertAlmostEqual(return_vals1['time'].loc[1][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2['time'].loc[1][18], 2.368, places=3)

    def test_covariance(self):
        from pyomo.contrib.interior_point.inverse_reduced_hessian import inv_reduced_hessian_barrier
        n = 60
        obj, theta, cov = self.pest_df.theta_est(calc_cov=True, cov_n=n)
        vars_list = [self.m_df.k1, self.m_df.k2]
        solve_result, inv_red_hes = inv_reduced_hessian_barrier(self.m_df, independent_variables=vars_list, tee=True)
        l = len(vars_list)
        cov_interior_point = 2 * obj / (n - l) * inv_red_hes
        cov_interior_point = pd.DataFrame(cov_interior_point, ['k1', 'k2'], ['k1', 'k2'])
        cov_diff = (cov - cov_interior_point).abs().sum().sum()
        self.assertTrue(cov.loc['k1', 'k1'] > 0)
        self.assertTrue(cov.loc['k2', 'k2'] > 0)
        self.assertAlmostEqual(cov_diff, 0, places=6)