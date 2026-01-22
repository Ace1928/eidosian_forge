import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _ter_compute(total_num_edits: Tensor, total_tgt_length: Tensor) -> Tensor:
    """Compute TER based on pre-computed a total number of edits and a total average reference length.

    Args:
        total_num_edits: A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length: A total average length of reference sentences.

    Return:
        A corpus-level TER score.

    """
    return _compute_ter_score_from_statistics(total_num_edits, total_tgt_length)