from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class IPv6NetworkError(PydanticValueError):
    msg_template = 'value is not a valid IPv6 network'