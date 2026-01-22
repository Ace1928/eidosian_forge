import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class InputToken:
    """
    Represents an input token.

    Args:
        id (`int`):
            Token ID from the model tokenizer.
        text (`str`):
            Token text.
        logprob (`float` or `None`):
            Log probability of the token. Optional since the logprob of the first token cannot be computed.
    """
    id: int
    text: str
    logprob: Optional[float] = None