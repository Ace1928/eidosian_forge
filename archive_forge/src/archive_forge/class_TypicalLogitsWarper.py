import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class TypicalLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens whose
    log probability is close to the entropy of the token probability distribution. This means that the most likely
    tokens may be discarded in the process.

    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`, *optional*, defaults to 0.9):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("1, 2, 3", return_tensors="pt")

    >>> # We can see that greedy decoding produces a sequence of numbers
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

    >>> # For this particular seed, we can see that sampling produces nearly the same low-information (= low entropy)
    >>> # sequence
    >>> set_seed(18)
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9 and 10

    >>> # With `typical_p` set, the most obvious sequence is no longer produced, which may be good for your problem
    >>> set_seed(18)
    >>> outputs = model.generate(
    ...     **inputs, do_sample=True, typical_p=0.1, return_dict_in_generate=True, output_scores=True
    ... )
    >>> print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])
    1, 2, 3 and 5

    >>> # We can see that the token corresponding to "4" (token 934) in the second position, the most likely token
    >>> # as seen with greedy decoding, was entirely blocked out
    >>> print(outputs.scores[1][0, 934])
    tensor(-inf)
    ```
    """

    def __init__(self, mass: float=0.9, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f'`typical_p` has to be a float > 0 and < 1, but is {mass}')
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(f'`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}')
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)
        shifted_scores = torch.abs(-normalized - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores