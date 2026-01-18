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
def theta_est(self, solver='ef_ipopt', return_values=[], calc_cov=False, cov_n=None):
    """
        Parameter estimation using all scenarios in the data

        Parameters
        ----------
        solver: string, optional
            Currently only "ef_ipopt" is supported. Default is "ef_ipopt".
        return_values: list, optional
            List of Variable names, used to return values from the model for data reconciliation
        calc_cov: boolean, optional
            If True, calculate and return the covariance matrix (only for "ef_ipopt" solver)
        cov_n: int, optional
            If calc_cov=True, then the user needs to supply the number of datapoints
            that are used in the objective function

        Returns
        -------
        objectiveval: float
            The objective function value
        thetavals: pd.Series
            Estimated values for theta
        variable values: pd.DataFrame
            Variable values for each variable name in return_values (only for solver='ef_ipopt')
        cov: pd.DataFrame
            Covariance matrix of the fitted parameters (only for solver='ef_ipopt')
        """
    assert isinstance(solver, str)
    assert isinstance(return_values, list)
    assert isinstance(calc_cov, bool)
    if calc_cov:
        assert isinstance(cov_n, int), 'The number of datapoints that are used in the objective function is required to calculate the covariance matrix'
        assert cov_n > len(self._return_theta_names()), 'The number of datapoints must be greater than the number of parameters to estimate'
    return self._Q_opt(solver=solver, return_values=return_values, bootlist=None, calc_cov=calc_cov, cov_n=cov_n)