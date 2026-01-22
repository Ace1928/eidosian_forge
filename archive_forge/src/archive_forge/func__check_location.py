import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
def _check_location(location: Optional[int], location_name: str, min_value: int) -> None:
    if location is None:
        return
    if not isinstance(location, int):
        raise TypeError(f'{location_name} must be an int, got {type(location)}')
    if location < min_value:
        raise ValueError(f'invalid {location_name}, expected >= {min_value}, got {location}')