import math
from typing import List, Optional, Tuple
import torch
def _gen_attention_probs(self, attention_weights: torch.Tensor, attention_mask: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    attention_weights_float = attention_weights.float()
    attention_weights_float = attention_weights_float.masked_fill(attention_mask.unsqueeze(0), self.negative_inf)
    T = attention_weights.size(1)
    B = attention_weights.size(0) // self.num_heads
    if padding_mask is not None:
        attention_weights_float = attention_weights_float.view(B, self.num_heads, T, -1)
        attention_weights_float = attention_weights_float.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), self.negative_inf)
        attention_weights_float = attention_weights_float.view(B * self.num_heads, T, -1)
    attention_probs = torch.nn.functional.softmax(attention_weights_float, dim=-1).type_as(attention_weights)
    return torch.nn.functional.dropout(attention_probs, p=float(self.dropout), training=self.training)