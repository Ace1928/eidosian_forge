from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_hypo_score(hypo: Hypothesis) -> float:
    return hypo[3]