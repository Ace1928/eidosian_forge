from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _batch_state(hypos: List[Hypothesis]) -> List[List[torch.Tensor]]:
    states: List[List[torch.Tensor]] = []
    for i in range(len(_get_hypo_state(hypos[0]))):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(_get_hypo_state(hypos[0])[i])):
            batched_state_components.append(torch.cat([_get_hypo_state(hypo)[i][j] for hypo in hypos]))
        states.append(batched_state_components)
    return states