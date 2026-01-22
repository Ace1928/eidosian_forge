import torch
from torch import Tensor
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
def _concordance_corrcoef_compute(mean_x: Tensor, mean_y: Tensor, var_x: Tensor, var_y: Tensor, corr_xy: Tensor, nb: Tensor) -> Tensor:
    """Compute the final concordance correlation coefficient based on accumulated statistics."""
    pearson = _pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
    return 2.0 * pearson * var_x.sqrt() * var_y.sqrt() / (var_x + var_y + (mean_x - mean_y) ** 2)