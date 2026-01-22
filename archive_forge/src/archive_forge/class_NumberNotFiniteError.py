from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class NumberNotFiniteError(PydanticValueError):
    code = 'number.not_finite_number'
    msg_template = 'ensure this value is a finite number'