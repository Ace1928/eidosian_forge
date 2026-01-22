import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
@dataclass(frozen=True)
class OpenAIConfig:
    """Represents the parameters of the OpenAI API.

    The information was last fetched on 2023/11/20. We document below the
    properties that are specific to the OpenAI API. Not all these properties are
    supported by Outlines.

    Properties
    ----------
    model
        The name of the model. Available models can be found on OpenAI's website.
    frequence_penalty
        Number between 2.0 and -2.0. Positive values penalize new tokens based on
        their existing frequency in the text,
    logit_bias
        Modifies the likelihood of specified tokens to appear in the completion.
        Number between -100 (forbid) and +100 (only allows).
    n
        The number of completions to return for each prompt.
    presence_penalty
        Similar to frequency penalty.
    response_format
        Specifies the format the model must output. `{"type": "json_object"}`
        enables JSON mode.
    seed
        Two completions with the same `seed` value should return the same
        completion. This is however not guaranteed.
    stop
        Up to 4 words where the API will stop the completion.
    temperature
        Number between 0 and 2. Higher values make the output more random, while
        lower values make it more deterministic.
    top_p
        Number between 0 and 1. Parameter for nucleus sampling.
    user
        A unique identifier for the end-user.

    """
    model: str = ''
    frequency_penalty: float = 0
    logit_bias: Dict[int, int] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: int = 1
    user: str = field(default_factory=str)