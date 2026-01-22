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
def _assert_replace(self, **kw: Any) -> URL:
    """argument checks before calling _replace()"""
    if 'drivername' in kw:
        self._assert_str(kw['drivername'], 'drivername')
    for name in ('username', 'host', 'database'):
        if name in kw:
            self._assert_none_str(kw[name], name)
    if 'port' in kw:
        self._assert_port(kw['port'])
    if 'query' in kw:
        kw['query'] = self._str_dict(kw['query'])
    return self._replace(**kw)