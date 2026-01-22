import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
def _format_summary_table(total_parameters: int, trainable_parameters: int, model_size: float, *cols: Tuple[str, List[str]]) -> str:
    """Takes in a number of arrays, each specifying a column in the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted."""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)
    col_widths = []
    for c in cols:
        col_width = max((len(str(a)) for a in c[1])) if n_rows else 0
        col_width = max(col_width, len(c[0]))
        col_widths.append(col_width)
    s = '{:<{}}'
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]
    summary = ' | '.join(header) + '\n' + '-' * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += '\n' + ' | '.join(line)
    summary += '\n' + '-' * total_width
    summary += '\n' + s.format(get_human_readable_count(trainable_parameters), 10)
    summary += 'Trainable params'
    summary += '\n' + s.format(get_human_readable_count(total_parameters - trainable_parameters), 10)
    summary += 'Non-trainable params'
    summary += '\n' + s.format(get_human_readable_count(total_parameters), 10)
    summary += 'Total params'
    summary += '\n' + s.format(get_formatted_model_size(model_size), 10)
    summary += 'Total estimated model params size (MB)'
    return summary