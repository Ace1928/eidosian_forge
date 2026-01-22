from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
@staticmethod
def _ignore_causal_mask_sdpa(attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor, past_key_values_length: int, sliding_window: Optional[int]=None) -> bool:
    """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
        """
    batch_size, query_length = (inputs_embeds.shape[0], inputs_embeds.shape[1])
    key_value_length = query_length + past_key_values_length
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy) or (hasattr(torch, '_dynamo') and torch._dynamo.is_compiling())
    ignore_causal_mask = False
    if attention_mask is None:
        if not is_tracing and (query_length == 1 or key_value_length == query_length) and (sliding_window is None or key_value_length < sliding_window):
            ignore_causal_mask = True
    elif sliding_window is None or key_value_length < sliding_window:
        if len(attention_mask.shape) == 4:
            expected_shape = (batch_size, 1, query_length, key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(f'Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}.')
        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1 or key_value_length == query_length:
                ignore_causal_mask = True
    return ignore_causal_mask