from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import compat
from .langhelpers import _hash_limit_string
from .langhelpers import _warnings_warn
from .langhelpers import decorator
from .langhelpers import inject_docstring_text
from .langhelpers import inject_param_text
from .. import exc
def warn_deprecated_limited(msg: str, args: Sequence[Any], version: str, stacklevel: int=3, code: Optional[str]=None) -> None:
    """Issue a deprecation warning with a parameterized string,
    limiting the number of registrations.

    """
    if args:
        msg = _hash_limit_string(msg, 10, args)
    _warn_with_version(msg, version, exc.SADeprecationWarning, stacklevel, code=code)