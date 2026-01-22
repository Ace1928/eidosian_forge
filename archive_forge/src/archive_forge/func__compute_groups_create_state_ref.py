from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import ModuleDict
from typing_extensions import Literal
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import _flatten_dict, allclose
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
def _compute_groups_create_state_ref(self, copy: bool=False) -> None:
    """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference

        """
    if not self._state_is_copy:
        for cg in self._groups.values():
            m0 = getattr(self, cg[0])
            for i in range(1, len(cg)):
                mi = getattr(self, cg[i])
                for state in m0._defaults:
                    m0_state = getattr(m0, state)
                    setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                mi._update_count = deepcopy(m0._update_count) if copy else m0._update_count
                mi._computed = deepcopy(m0._computed) if copy else m0._computed
    self._state_is_copy = copy