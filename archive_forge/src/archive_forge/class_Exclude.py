import functools
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar, Union
from marshmallow.fields import Field as MarshmallowField  # type: ignore
from dataclasses_json.stringcase import (camelcase, pascalcase, snakecase,
from dataclasses_json.undefined import Undefined, UndefinedParameterError
class Exclude:
    """
    Pre-defined constants for exclusion. By default, fields are configured to
    be included.
    """
    ALWAYS: Callable[[object], bool] = lambda _: True
    NEVER: Callable[[object], bool] = lambda _: False