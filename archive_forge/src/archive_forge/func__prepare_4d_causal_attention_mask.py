from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
def _prepare_4d_causal_attention_mask(attention_mask: Optional[torch.Tensor], input_shape: Union[torch.Size, Tuple, List], inputs_embeds: torch.Tensor, past_key_values_length: int, sliding_window: Optional[int]=None):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype)
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(f'Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}.')
        else:
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
    else:
        attention_mask = attn_mask_converter.to_causal_4d(input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return attention_mask