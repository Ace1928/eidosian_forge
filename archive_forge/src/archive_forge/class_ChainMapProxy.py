import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
class ChainMapProxy(Mapping[Union[str, AppKey[Any]], Any]):
    __slots__ = ('_maps',)

    def __init__(self, maps: Iterable[Mapping[Union[str, AppKey[Any]], Any]]) -> None:
        self._maps = tuple(maps)

    def __init_subclass__(cls) -> None:
        raise TypeError('Inheritance class {} from ChainMapProxy is forbidden'.format(cls.__name__))

    @overload
    def __getitem__(self, key: AppKey[_T]) -> _T:
        ...

    @overload
    def __getitem__(self, key: str) -> Any:
        ...

    def __getitem__(self, key: Union[str, AppKey[_T]]) -> Any:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    @overload
    def get(self, key: AppKey[_T], default: _S) -> Union[_T, _S]:
        ...

    @overload
    def get(self, key: AppKey[_T], default: None=...) -> Optional[_T]:
        ...

    @overload
    def get(self, key: str, default: Any=...) -> Any:
        ...

    def get(self, key: Union[str, AppKey[_T]], default: Any=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(set().union(*self._maps))

    def __iter__(self) -> Iterator[Union[str, AppKey[Any]]]:
        d: Dict[Union[str, AppKey[Any]], Any] = {}
        for mapping in reversed(self._maps):
            d.update(mapping)
        return iter(d)

    def __contains__(self, key: object) -> bool:
        return any((key in m for m in self._maps))

    def __bool__(self) -> bool:
        return any(self._maps)

    def __repr__(self) -> str:
        content = ', '.join(map(repr, self._maps))
        return f'ChainMapProxy({content})'