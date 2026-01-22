import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class TextGenerationRequest:
    """
    Request object for text generation (only for internal use).

    Args:
        inputs (`str`):
            The prompt for text generation.
        parameters (`Optional[TextGenerationParameters]`, *optional*):
            Generation parameters.
        stream (`bool`, *optional*):
            Whether to stream output tokens. Defaults to False.
    """
    inputs: str
    parameters: Optional[TextGenerationParameters] = None
    stream: bool = False

    @validator('inputs')
    def valid_input(cls, v):
        if not v:
            raise ValueError('`inputs` cannot be empty')
        return v

    @validator('stream')
    def valid_best_of_stream(cls, field_value, values):
        parameters = values['parameters']
        if parameters is not None and parameters.best_of is not None and (parameters.best_of > 1) and field_value:
            raise ValueError('`best_of` != 1 is not supported when `stream` == True')
        return field_value

    def __post_init__(self):
        if not is_pydantic_available():
            if self.parameters is not None and isinstance(self.parameters, dict):
                self.parameters = TextGenerationParameters(**self.parameters)