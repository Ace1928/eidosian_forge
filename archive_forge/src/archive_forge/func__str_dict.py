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
@classmethod
def _str_dict(cls, dict_: Optional[Union[Sequence[Tuple[str, Union[Sequence[str], str]]], Mapping[str, Union[Sequence[str], str]]]]) -> util.immutabledict[str, Union[Tuple[str, ...], str]]:
    if dict_ is None:
        return util.EMPTY_DICT

    @overload
    def _assert_value(val: str) -> str:
        ...

    @overload
    def _assert_value(val: Sequence[str]) -> Union[str, Tuple[str, ...]]:
        ...

    def _assert_value(val: Union[str, Sequence[str]]) -> Union[str, Tuple[str, ...]]:
        if isinstance(val, str):
            return val
        elif isinstance(val, collections_abc.Sequence):
            return tuple((_assert_value(elem) for elem in val))
        else:
            raise TypeError('Query dictionary values must be strings or sequences of strings')

    def _assert_str(v: str) -> str:
        if not isinstance(v, str):
            raise TypeError('Query dictionary keys must be strings')
        return v
    dict_items: Iterable[Tuple[str, Union[Sequence[str], str]]]
    if isinstance(dict_, collections_abc.Sequence):
        dict_items = dict_
    else:
        dict_items = dict_.items()
    return util.immutabledict({_assert_str(key): _assert_value(value) for key, value in dict_items})