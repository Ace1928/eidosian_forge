from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class PatternError(PydanticValueError):
    code = 'regex_pattern'
    msg_template = 'Invalid regular expression'