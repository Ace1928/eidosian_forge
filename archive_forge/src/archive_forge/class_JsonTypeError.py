from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class JsonTypeError(PydanticTypeError):
    code = 'json'
    msg_template = 'JSON object must be str, bytes or bytearray'