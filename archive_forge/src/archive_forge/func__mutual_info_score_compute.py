import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels
def _mutual_info_score_compute(contingency: Tensor) -> Tensor:
    """Compute the mutual information score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        mutual_info: mutual information score

    """
    n = contingency.sum()
    u = contingency.sum(dim=1)
    v = contingency.sum(dim=0)
    if u.size() == 1 or v.size() == 1:
        return tensor(0.0)
    nzu, nzv = torch.nonzero(contingency, as_tuple=True)
    contingency = contingency[nzu, nzv]
    log_outer = torch.log(u[nzu]) + torch.log(v[nzv])
    mutual_info = contingency / n * (torch.log(n) + torch.log(contingency) - log_outer)
    return mutual_info.sum()