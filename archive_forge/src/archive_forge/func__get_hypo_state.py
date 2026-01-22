from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_hypo_state(hypo: Hypothesis) -> List[List[torch.Tensor]]:
    return hypo[2]