import sys
from os import PathLike
from typing import TYPE_CHECKING
from typing import List, Dict, AnyStr, Any, Set
from typing import Optional, Union, Tuple, Mapping, Sequence, TypeVar, Type
from typing import Callable, Coroutine, Generator, AsyncGenerator, IO, Iterable, Iterator, AsyncIterator
from typing import cast, overload
from enum import Enum, EnumMeta
from functools import singledispatchmethod
class ListStr(list):
    """
        Returns List[str] by splitting on delimiter ','
        """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Optional[Any]) -> Optional[List[str]]:
        """
            Validates the value and returns a List[str]
            """
        if v is None:
            return None
        from lazyops.utils.serialization import parse_list_str
        return parse_list_str(value)