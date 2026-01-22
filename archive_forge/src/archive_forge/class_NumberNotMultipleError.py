from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class NumberNotMultipleError(PydanticValueError):
    code = 'number.not_multiple'
    msg_template = 'ensure this value is a multiple of {multiple_of}'

    def __init__(self, *, multiple_of: Union[int, float, Decimal]) -> None:
        super().__init__(multiple_of=multiple_of)