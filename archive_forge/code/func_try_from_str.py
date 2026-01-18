import warnings
from enum import Enum
from typing import List, Optional
from typing_extensions import Literal
@classmethod
def try_from_str(cls, value: str, source: Literal['key', 'value', 'any']='key') -> Optional['StrEnum']:
    """Try to create emun and if it does not match any, return `None`."""
    try:
        return cls.from_str(value, source)
    except ValueError:
        warnings.warn(UserWarning(f'Invalid string: expected one of {cls._allowed_matches(source)}, but got {value}.'))
    return None