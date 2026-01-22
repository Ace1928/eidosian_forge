from typing import TYPE_CHECKING, Optional
import torch
from .transformers import TransformerTokenizer
class ExLlamaV2Model:
    """Represents a `exl2` model."""

    def __init__(self, model: 'ExLlamaV2', tokenizer: 'PreTrainedTokenizer', device, cache: 'ExLlamaV2Cache'):
        self.device = device
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)
        self.cache = cache
        self.past_seq = None

    def forward(self, input_ids: torch.LongTensor, *_):
        """Compute a forward pass through the exl2 model."""
        reset = True
        seq_tensor = input_ids[0]
        if self.past_seq is not None:
            min_length = min(self.past_seq.shape[0], seq_tensor.shape[0])
            indices = torch.nonzero(~torch.eq(self.past_seq[:min_length], seq_tensor[:min_length]))
            if len(indices) > 0:
                longest_prefix = indices[0].item()
            else:
                longest_prefix = min_length
            if longest_prefix > 0:
                reset = False
                self.cache.current_seq_len = longest_prefix
                if seq_tensor.shape[0] - longest_prefix > 1:
                    self.model.forward(seq_tensor[longest_prefix:-1].view(1, -1), self.cache, preprocess_only=True)
                elif seq_tensor.shape[0] == longest_prefix:
                    self.cache.current_seq_len -= 1
        if reset:
            self.cache.current_seq_len = 0
            if seq_tensor.shape[0] > 1:
                self.model.forward(seq_tensor[:-1].view(1, -1), self.cache, preprocess_only=True)
        self.past_seq = seq_tensor
        return self.model.forward(seq_tensor[-1:].view(1, -1), self.cache)

    def __call__(self, input_ids: torch.LongTensor, *_) -> torch.FloatTensor:
        logits = self.forward(input_ids)
        next_token_logits = logits[..., -1, :]
        return (next_token_logits, None)