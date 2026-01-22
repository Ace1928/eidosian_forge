import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
def _get_required(d: Dict[str, Any], expected_type: Type[T], key: str, default: Optional[T]=None) -> T:
    value = _get(d, expected_type, key, default)
    if value is None:
        raise DirectUrlValidationError(f'{key} must have a value')
    return value