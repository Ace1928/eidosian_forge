import abc
import io
import itertools
import pathlib
from typing import Any, BinaryIO, Iterable, Iterator, NoReturn, Text, Optional
from ._compat import runtime_checkable, Protocol, StrPath
class TraversalError(Exception):
    pass