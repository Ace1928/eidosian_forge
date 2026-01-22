from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class PathNotExistsError(_PathValueError):
    code = 'path.not_exists'
    msg_template = 'file or directory at path "{path}" does not exist'