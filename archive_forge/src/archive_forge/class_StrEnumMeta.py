import sys
from os import PathLike
from typing import TYPE_CHECKING
from typing import List, Dict, AnyStr, Any, Set
from typing import Optional, Union, Tuple, Mapping, Sequence, TypeVar, Type
from typing import Callable, Coroutine, Generator, AsyncGenerator, IO, Iterable, Iterator, AsyncIterator
from typing import cast, overload
from enum import Enum, EnumMeta
from functools import singledispatchmethod
class StrEnumMeta(EnumMeta):

    @singledispatchmethod
    def __getitem__(self, key):
        return super().__getitem__(key)

    @__getitem__.register
    def _(self, index: int):
        return list(self)[index]