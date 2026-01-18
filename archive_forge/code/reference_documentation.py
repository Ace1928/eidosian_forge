from git.util import (
from .symbolic import SymbolicReference, T_References
from typing import Any, Callable, Iterator, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike, _T

        :return: Name of the remote head itself, e.g. master.

        :note: The returned name is usually not qualified enough to uniquely identify
            a branch.
        