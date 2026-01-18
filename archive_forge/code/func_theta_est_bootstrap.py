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
def theta_est_bootstrap(self, bootstrap_samples, samplesize=None, replacement=True, seed=None, return_samples=False):
    """
        Parameter estimation using bootstrap resampling of the data

        Parameters
        ----------
        bootstrap_samples: int
            Number of bootstrap samples to draw from the data
        samplesize: int or None, optional
            Size of each bootstrap sample. If samplesize=None, samplesize will be
            set to the number of samples in the data
        replacement: bool, optional
            Sample with or without replacement
        seed: int or None, optional
            Random seed
        return_samples: bool, optional
            Return a list of sample numbers used in each bootstrap estimation

        Returns
        -------
        bootstrap_theta: pd.DataFrame
            Theta values for each sample and (if return_samples = True)
            the sample numbers used in each estimation
        """
    assert isinstance(bootstrap_samples, int)
    assert isinstance(samplesize, (type(None), int))
    assert isinstance(replacement, bool)
    assert isinstance(seed, (type(None), int))
    assert isinstance(return_samples, bool)
    if samplesize is None:
        samplesize = len(self.callback_data)
    if seed is not None:
        np.random.seed(seed)
    global_list = self._get_sample_list(samplesize, bootstrap_samples, replacement)
    task_mgr = utils.ParallelTaskManager(bootstrap_samples)
    local_list = task_mgr.global_to_local_data(global_list)
    bootstrap_theta = list()
    for idx, sample in local_list:
        objval, thetavals = self._Q_opt(bootlist=list(sample))
        thetavals['samples'] = sample
        bootstrap_theta.append(thetavals)
    global_bootstrap_theta = task_mgr.allgather_global_data(bootstrap_theta)
    bootstrap_theta = pd.DataFrame(global_bootstrap_theta)
    if not return_samples:
        del bootstrap_theta['samples']
    return bootstrap_theta