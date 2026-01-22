from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class IPvAnyAddressError(PydanticValueError):
    msg_template = 'value is not a valid IPv4 or IPv6 address'