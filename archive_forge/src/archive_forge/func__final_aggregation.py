from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _final_aggregation(means_x: Tensor, means_y: Tensor, vars_x: Tensor, vars_y: Tensor, corrs_xy: Tensor, nbs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Aggregate the statistics from multiple devices`_

    """
    if len(means_x) == 1:
        return (means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0])
    mx1, my1, vx1, vy1, cxy1, n1 = (means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0])
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = (means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i])
        nb = n1 + n2
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb
        element_x1 = (n1 + 1) * mean_x - n1 * mx1
        vx1 += (element_x1 - mx1) * (element_x1 - mean_x) - (element_x1 - mean_x) ** 2
        element_x2 = (n2 + 1) * mean_x - n2 * mx2
        vx2 += (element_x2 - mx2) * (element_x2 - mean_x) - (element_x2 - mean_x) ** 2
        var_x = vx1 + vx2
        element_y1 = (n1 + 1) * mean_y - n1 * my1
        vy1 += (element_y1 - my1) * (element_y1 - mean_y) - (element_y1 - mean_y) ** 2
        element_y2 = (n2 + 1) * mean_y - n2 * my2
        vy2 += (element_y2 - my2) * (element_y2 - mean_y) - (element_y2 - mean_y) ** 2
        var_y = vy1 + vy2
        cxy1 += (element_x1 - mx1) * (element_y1 - mean_y) - (element_x1 - mean_x) * (element_y1 - mean_y)
        cxy2 += (element_x2 - mx2) * (element_y2 - mean_y) - (element_x2 - mean_x) * (element_y2 - mean_y)
        corr_xy = cxy1 + cxy2
        mx1, my1, vx1, vy1, cxy1, n1 = (mean_x, mean_y, var_x, var_y, corr_xy, nb)
    return (mean_x, mean_y, var_x, var_y, corr_xy, nb)