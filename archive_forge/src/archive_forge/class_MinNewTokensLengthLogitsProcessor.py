import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["A number:"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_new_tokens` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_new_tokens=2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand
    ```
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int]]):
        for arg_name, arg_value in [('prompt_length_to_skip', prompt_length_to_skip), ('min_new_tokens', min_new_tokens)]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f'`{arg_name}` has to be a positive integer, but is {arg_value}')
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all((isinstance(i, int) for i in eos_token_id)) or any((i < 0 for i in eos_token_id)):
            logger.warning(f'`eos_token_id` has to be a list of positive integers, but is {eos_token_id}')
        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float('inf')
        return scores