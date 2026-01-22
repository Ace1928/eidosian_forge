from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def _assert_value(val: Union[str, Sequence[str]]) -> Union[str, Tuple[str, ...]]:
    if isinstance(val, str):
        return val
    elif isinstance(val, collections_abc.Sequence):
        return tuple((_assert_value(elem) for elem in val))
    else:
        raise TypeError('Query dictionary values must be strings or sequences of strings')