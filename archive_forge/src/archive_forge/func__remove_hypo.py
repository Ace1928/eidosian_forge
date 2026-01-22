from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _remove_hypo(hypo: Hypothesis, hypo_list: List[Hypothesis]) -> None:
    for i, elem in enumerate(hypo_list):
        if _get_hypo_key(hypo) == _get_hypo_key(elem):
            del hypo_list[i]
            break