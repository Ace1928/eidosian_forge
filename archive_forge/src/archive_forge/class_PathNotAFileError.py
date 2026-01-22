from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class PathNotAFileError(_PathValueError):
    code = 'path.not_a_file'
    msg_template = 'path "{path}" does not point to a file'