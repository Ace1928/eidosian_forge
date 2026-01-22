import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class ExponentialDecayLengthPenalty(LogitsProcessor):
    """
    [`LogitsProcessor`] that exponentially increases the score of the `eos_token_id` after `start_index` has been
    reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
    predicted in a meaningful position.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        input_ids_seq_length (`int`):
            The length of the input sequence.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    >>> text = "Just wanted to let you know, I"
    >>> inputs = tokenizer(text, return_tensors="pt")

    >>> # Let's consider that we want short sentences, so we limit `max_length=30`. However, we observe that the answer
    >>> # tends to end abruptly.
    >>> set_seed(1)
    >>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010. Although

    >>> # To promote the appearance of the EOS token at the right time, we add the `exponential_decay_length_penalty =
    >>> # (start_index, decay_factor)`. Instead of cutting at max_tokens, the output comes to an end before and usually
    >>> # with more meaning. What happens is that starting from `start_index` the EOS token score will be increased
    >>> # by `decay_factor` exponentially. However, if you set a high decay factor, you may also end up with abruptly
    >>> # ending sequences.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.6),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network
    which<|endoftext|>

    >>> # With a small decay factor, you will have a higher chance of getting a meaningful sequence.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.01),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010.<|endoftext|>
    ```
    """

    def __init__(self, exponential_decay_length_penalty: Tuple[int, float], eos_token_id: Union[int, List[int]], input_ids_seq_length: int):
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            for i in self.eos_token_id:
                penalty_idx = cur_len - self.regulation_start
                scores[:, i] = scores[:, i] + torch.abs(scores[:, i]) * (pow(self.regulation_factor, penalty_idx) - 1)
        return scores