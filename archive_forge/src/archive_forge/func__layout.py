import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def _layout(self, results: List[common.Measurement]):
    table = Table(results, self._colorize, self._trim_significant_figures, self._highlight_warnings)
    return table.render()