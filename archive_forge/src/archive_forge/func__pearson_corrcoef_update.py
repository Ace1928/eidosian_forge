import math
from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
def _pearson_corrcoef_update(preds: Tensor, target: Tensor, mean_x: Tensor, mean_y: Tensor, var_x: Tensor, var_y: Tensor, corr_xy: Tensor, num_prior: Tensor, num_outputs: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        num_prior: current number of observed observations
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    num_obs = preds.shape[0]
    cond = num_prior.mean() > 0 or num_obs == 1
    if cond:
        mx_new = (num_prior * mean_x + preds.sum(0)) / (num_prior + num_obs)
        my_new = (num_prior * mean_y + target.sum(0)) / (num_prior + num_obs)
    else:
        mx_new = preds.mean(0).to(mean_x.dtype)
        my_new = target.mean(0).to(mean_y.dtype)
    num_prior += num_obs
    if cond:
        var_x += ((preds - mx_new) * (preds - mean_x)).sum(0)
        var_y += ((target - my_new) * (target - mean_y)).sum(0)
    else:
        var_x += preds.var(0) * (num_obs - 1)
        var_y += target.var(0) * (num_obs - 1)
    corr_xy += ((preds - mx_new) * (target - mean_y)).sum(0)
    mean_x = mx_new
    mean_y = my_new
    return (mean_x, mean_y, var_x, var_y, corr_xy, num_prior)