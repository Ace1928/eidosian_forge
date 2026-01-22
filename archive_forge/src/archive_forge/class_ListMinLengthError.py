from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class ListMinLengthError(PydanticValueError):
    code = 'list.min_items'
    msg_template = 'ensure this value has at least {limit_value} items'

    def __init__(self, *, limit_value: int) -> None:
        super().__init__(limit_value=limit_value)