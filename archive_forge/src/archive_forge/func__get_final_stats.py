from typing import Any, Callable, Optional, Sequence, Tuple, Union, no_type_check
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.dice import _dice_compute
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
@no_type_check
def _get_final_stats(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform concatenation on the stat scores if necessary, before passing them to a compute function."""
    tp = torch.cat(self.tp) if isinstance(self.tp, list) else self.tp
    fp = torch.cat(self.fp) if isinstance(self.fp, list) else self.fp
    tn = torch.cat(self.tn) if isinstance(self.tn, list) else self.tn
    fn = torch.cat(self.fn) if isinstance(self.fn, list) else self.fn
    return (tp, fp, tn, fn)