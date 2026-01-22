from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class InvalidByteSize(PydanticValueError):
    msg_template = 'could not parse value and unit from byte string'