import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _Q_opt(self, ThetaVals=None, solver='ef_ipopt', return_values=[], bootlist=None, calc_cov=False, cov_n=None):
    """
        Set up all thetas as first stage Vars, return resulting theta
        values as well as the objective function value.

        """
    if solver == 'k_aug':
        raise RuntimeError('k_aug no longer supported.')
    if bootlist is None:
        scenario_numbers = list(range(len(self.callback_data)))
        scen_names = ['Scenario{}'.format(i) for i in scenario_numbers]
    else:
        scen_names = ['Scenario{}'.format(i) for i in range(len(bootlist))]
    outer_cb_data = dict()
    outer_cb_data['callback'] = self._instance_creation_callback
    if ThetaVals is not None:
        outer_cb_data['ThetaVals'] = ThetaVals
    if bootlist is not None:
        outer_cb_data['BootList'] = bootlist
    outer_cb_data['cb_data'] = self.callback_data
    outer_cb_data['theta_names'] = self.theta_names
    options = {'solver': 'ipopt'}
    scenario_creator_options = {'cb_data': outer_cb_data}
    if use_mpisppy:
        ef = sputils.create_EF(scen_names, _experiment_instance_creation_callback, EF_name='_Q_opt', suppress_warnings=True, scenario_creator_kwargs=scenario_creator_options)
    else:
        ef = local_ef.create_EF(scen_names, _experiment_instance_creation_callback, EF_name='_Q_opt', suppress_warnings=True, scenario_creator_kwargs=scenario_creator_options)
    self.ef_instance = ef
    if solver == 'ef_ipopt':
        if not calc_cov:
            solver = SolverFactory('ipopt')
            if self.solver_options is not None:
                for key in self.solver_options:
                    solver.options[key] = self.solver_options[key]
            solve_result = solver.solve(self.ef_instance, tee=self.tee)
        else:
            ind_vars = []
            for ndname, Var, solval in ef_nonants(ef):
                ind_vars.append(Var)
            solve_result, inv_red_hes = inverse_reduced_hessian.inv_reduced_hessian_barrier(self.ef_instance, independent_variables=ind_vars, solver_options=self.solver_options, tee=self.tee)
        if self.diagnostic_mode:
            print('    Solver termination condition = ', str(solve_result.solver.termination_condition))
        thetavals = {}
        for ndname, Var, solval in ef_nonants(ef):
            vname = Var.name[Var.name.find('.') + 1:]
            thetavals[vname] = solval
        objval = pyo.value(ef.EF_Obj)
        if calc_cov:
            n = cov_n
            l = len(thetavals)
            sse = objval
            'Calculate covariance assuming experimental observation errors are\n                independent and follow a Gaussian\n                distribution with constant variance.\n\n                The formula used in parmest was verified against equations (7-5-15) and\n                (7-5-16) in "Nonlinear Parameter Estimation", Y. Bard, 1974.\n\n                This formula is also applicable if the objective is scaled by a constant;\n                the constant cancels out. (was scaled by 1/n because it computes an\n                expected value.)\n                '
            cov = 2 * sse / (n - l) * inv_red_hes
            cov = pd.DataFrame(cov, index=thetavals.keys(), columns=thetavals.keys())
        thetavals = pd.Series(thetavals)
        if len(return_values) > 0:
            var_values = []
            if len(scen_names) > 1:
                block_objects = self.ef_instance.component_objects(Block, descend_into=False)
            else:
                block_objects = [self.ef_instance]
            for exp_i in block_objects:
                vals = {}
                for var in return_values:
                    exp_i_var = exp_i.find_component(str(var))
                    if exp_i_var is None:
                        continue
                    if type(exp_i_var) == ContinuousSet:
                        temp = list(exp_i_var)
                    else:
                        temp = [pyo.value(_) for _ in exp_i_var.values()]
                    if len(temp) == 1:
                        vals[var] = temp[0]
                    else:
                        vals[var] = temp
                if len(vals) > 0:
                    var_values.append(vals)
            var_values = pd.DataFrame(var_values)
            if calc_cov:
                return (objval, thetavals, var_values, cov)
            else:
                return (objval, thetavals, var_values)
        if calc_cov:
            return (objval, thetavals, cov)
        else:
            return (objval, thetavals)
    else:
        raise RuntimeError('Unknown solver in Q_Opt=' + solver)