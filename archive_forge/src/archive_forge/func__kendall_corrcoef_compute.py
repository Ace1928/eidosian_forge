from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _kendall_corrcoef_compute(preds: Tensor, target: Tensor, variant: _MetricVariant, alternative: Optional[_TestAlternative]=None) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test.

    Args:
        Args:
        preds: Sequence of data
        target: Sequence of data
        variant: Indication of which variant of Kendall's tau to be used
        alternative: Alternative hypothesis for for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    """
    concordant_pairs, discordant_pairs, preds_ties, preds_ties_p1, preds_ties_p2, target_ties, target_ties_p1, target_ties_p2, n_total = _get_metric_metadata(preds, target, variant)
    con_min_dis_pairs = concordant_pairs - discordant_pairs
    tau = _calculate_tau(preds, target, concordant_pairs, discordant_pairs, con_min_dis_pairs, n_total, preds_ties, target_ties, variant)
    p_value = _calculate_p_value(con_min_dis_pairs, n_total, preds_ties, preds_ties_p1, preds_ties_p2, target_ties, target_ties_p1, target_ties_p2, variant, alternative) if alternative else None
    if tau.shape[0] == 1:
        tau = tau.squeeze()
        p_value = p_value.squeeze() if p_value is not None else None
    return (tau.clamp(-1, 1), p_value)