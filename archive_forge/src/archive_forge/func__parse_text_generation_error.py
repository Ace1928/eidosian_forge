import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
def _parse_text_generation_error(error: Optional[str], error_type: Optional[str]) -> TextGenerationError:
    if error_type == 'generation':
        return GenerationError(error)
    if error_type == 'incomplete_generation':
        return IncompleteGenerationError(error)
    if error_type == 'overloaded':
        return OverloadedError(error)
    if error_type == 'validation':
        return ValidationError(error)
    return UnknownError(error)