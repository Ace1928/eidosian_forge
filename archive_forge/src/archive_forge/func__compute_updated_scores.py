from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _compute_updated_scores(hypos: List[Hypothesis], next_token_probs: torch.Tensor, beam_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hypo_scores = torch.tensor([_get_hypo_score(h) for h in hypos]).unsqueeze(1)
    nonblank_scores = hypo_scores + next_token_probs[:, :-1]
    nonblank_nbest_scores, nonblank_nbest_idx = nonblank_scores.reshape(-1).topk(beam_width)
    nonblank_nbest_hypo_idx = nonblank_nbest_idx.div(nonblank_scores.shape[1], rounding_mode='trunc')
    nonblank_nbest_token = nonblank_nbest_idx % nonblank_scores.shape[1]
    return (nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token)