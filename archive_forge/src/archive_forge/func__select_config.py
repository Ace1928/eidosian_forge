from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict
from ray.util.debug import log_once
def _select_config(Xraw: np.array, yraw: np.array, current: list, newpoint: np.array, bounds: dict, num_f: int) -> np.ndarray:
    """Selects the next hyperparameter config to try.

    This function takes the formatted data, fits the GP model and optimizes the
    UCB acquisition function to select the next point.

    Args:
        Xraw: The un-normalized array of hyperparams, Time and
            Reward
        yraw: The un-normalized vector of reward changes.
        current: The hyperparams of trials currently running. This is
            important so we do not select the same config twice. If there is
            data here then we fit a second GP including it
            (with fake y labels). The GP variance doesn't depend on the y
            labels so it is ok.
        newpoint: The Reward and Time for the new point.
            We cannot change these as they are based on the *new weights*.
        bounds: Bounds for the hyperparameters. Used to normalize.
        num_f: The number of fixed params. Almost always 2 (reward+time)

    Return:
        xt: A vector of new hyperparameters.
    """
    length = select_length(Xraw, yraw, bounds, num_f)
    Xraw = Xraw[-length:, :]
    yraw = yraw[-length:]
    base_vals = np.array(list(bounds.values())).T
    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate((np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))).reshape(2, oldpoints.shape[1])
    limits = np.concatenate((old_lims, base_vals), axis=1)
    X = normalize(Xraw, limits)
    y = standardize(yraw).reshape(yraw.size, 1)
    fixed = normalize(newpoint, oldpoints)
    kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
    try:
        m = GPy.models.GPRegression(X, y, kernel)
    except np.linalg.LinAlgError:
        X += np.eye(X.shape[0]) * 0.001
        m = GPy.models.GPRegression(X, y, kernel)
    try:
        m.optimize()
    except np.linalg.LinAlgError:
        X += np.eye(X.shape[0]) * 0.001
        m = GPy.models.GPRegression(X, y, kernel)
        m.optimize()
    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-05, 1))
    if current is None:
        m1 = deepcopy(m)
    else:
        padding = np.array([fixed for _ in range(current.shape[0])])
        current = normalize(current, base_vals)
        current = np.hstack((padding, current))
        Xnew = np.vstack((X, current))
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1, 1)
        ynew = np.vstack((y, ypad))
        kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
        m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
        m1.optimize()
    xt = optimize_acq(UCB, m, m1, fixed, num_f)
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(base_vals, axis=0)
    xt = xt.astype(np.float32)
    return xt