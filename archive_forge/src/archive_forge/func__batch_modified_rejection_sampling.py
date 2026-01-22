from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _batch_modified_rejection_sampling(self, target_probs: torch.Tensor, draft_probs: torch.Tensor, draft_token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """
    batch_size, k, vocab_size = draft_probs.shape
    accepted = self._get_accepted(target_probs, draft_probs, draft_token_ids)
    recovered_probs = self._get_recovered_probs(target_probs, draft_probs).reshape(batch_size * k, vocab_size)
    recovered_token_ids = _multinomial(recovered_probs, num_samples=1).reshape(batch_size, k)
    return (accepted, recovered_token_ids)