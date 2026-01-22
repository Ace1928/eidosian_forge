import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class TextGenerationResponse:
    """
    Represents a response for text generation.

    Only returned when `details=True`, otherwise a string is returned.

    Args:
        generated_text (`str`):
            The generated text.
        details (`Optional[Details]`):
            Generation details. Returned only if `details=True` is sent to the server.
    """
    generated_text: str
    details: Optional[Details] = None

    def __post_init__(self):
        if not is_pydantic_available():
            if self.details is not None and isinstance(self.details, dict):
                self.details = Details(**self.details)