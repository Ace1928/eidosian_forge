from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.group_fairness import (
from torchmetrics.functional.classification.stat_scores import _binary_stat_scores_arg_validation
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _update_states(self, group_stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
    for group, stats in enumerate(group_stats):
        tp, fp, tn, fn = stats
        self.tp[group] += tp
        self.fp[group] += fp
        self.tn[group] += tn
        self.fn[group] += fn