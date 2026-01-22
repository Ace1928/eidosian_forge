from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _gen_a_hypos(self, a_hypos: List[Hypothesis], b_hypos: List[Hypothesis], next_token_probs: torch.Tensor, t: int, beam_width: int, device: torch.device) -> List[Hypothesis]:
    nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token = _compute_updated_scores(a_hypos, next_token_probs, beam_width)
    if len(b_hypos) < beam_width:
        b_nbest_score = -float('inf')
    else:
        b_nbest_score = _get_hypo_score(b_hypos[-beam_width])
    base_hypos: List[Hypothesis] = []
    new_tokens: List[int] = []
    new_scores: List[float] = []
    for i in range(beam_width):
        score = float(nonblank_nbest_scores[i])
        if score > b_nbest_score:
            a_hypo_idx = int(nonblank_nbest_hypo_idx[i])
            base_hypos.append(a_hypos[a_hypo_idx])
            new_tokens.append(int(nonblank_nbest_token[i]))
            new_scores.append(score)
    if base_hypos:
        new_hypos = self._gen_new_hypos(base_hypos, new_tokens, new_scores, t, device)
    else:
        new_hypos: List[Hypothesis] = []
    return new_hypos