from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class BoolError(PydanticTypeError):
    msg_template = 'value could not be parsed to a boolean'