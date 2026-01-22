import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    """
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the begining. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
    >>> # it can't generate and EOS token in the first iteration, but it can in the others.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[1][0, 50256])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)
    >>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
    tensor(29.9010)

    >>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
    >>> outputs = model.generate(
    ...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    ... )
    >>> print(outputs.scores[1][0, 50256])
    tensor(11.2027)
    ```
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == self.begin_index:
            scores[:, self.begin_suppress_tokens] = -float('inf')
        return scores