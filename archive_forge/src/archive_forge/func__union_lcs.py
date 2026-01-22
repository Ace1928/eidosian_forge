import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _union_lcs(pred_tokens_list: Sequence[Sequence[str]], target_tokens: Sequence[str]) -> Sequence[str]:
    """Find union LCS between a target sentence and iterable of predicted tokens.

    Args:
        pred_tokens_list: A tokenized predicted sentence split by ``'\\n'``.
        target_tokens: A tokenized single part of target sentence split by ``'\\n'``.

    """

    def lcs_ind(pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> Sequence[int]:
        """Return one of the longest of longest common subsequence via backtracked lcs table."""
        lcs_table: Sequence[Sequence[int]] = _lcs(pred_tokens, target_tokens, return_full_table=True)
        return _backtracked_lcs(lcs_table, pred_tokens, target_tokens)

    def find_union(lcs_tables: Sequence[Sequence[int]]) -> Sequence[int]:
        """Find union LCS given a list of LCS."""
        return sorted(set().union(*lcs_tables))
    lcs_tables = [lcs_ind(pred_tokens, target_tokens) for pred_tokens in pred_tokens_list]
    return [target_tokens[i] for i in find_union(lcs_tables)]