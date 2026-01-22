import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class StreamDetails:
    """
    Represents details of a text generation stream.

    Args:
        finish_reason (`FinishReason`):
            The reason for the generation to finish, represented by a `FinishReason` value.
        generated_tokens (`int`):
            The number of generated tokens.
        seed (`Optional[int]`):
            The sampling seed if sampling was activated.
    """
    finish_reason: FinishReason
    generated_tokens: int
    seed: Optional[int] = None