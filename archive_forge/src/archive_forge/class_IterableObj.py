from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
@runtime_checkable
class IterableObj(Protocol):
    """Defines an interface for iterable items, so there is a uniform way to retrieve
    and iterate items within the git repository.

    Subclasses = [Submodule, Commit, Reference, PushInfo, FetchInfo, Remote]
    """
    __slots__ = ()
    _id_attribute_: str

    @classmethod
    @abstractmethod
    def iter_items(cls, repo: 'Repo', *args: Any, **kwargs: Any) -> Iterator[T_IterableObj]:
        """Find (all) items of this type.

        Subclasses can specify ``args`` and ``kwargs`` differently, and may use them for
        filtering. However, when the method is called with no additional positional or
        keyword arguments, subclasses are obliged to to yield all items.

        :return: Iterator yielding Items
        """
        raise NotImplementedError('To be implemented by Subclass')

    @classmethod
    def list_items(cls, repo: 'Repo', *args: Any, **kwargs: Any) -> IterableList[T_IterableObj]:
        """Find (all) items of this type and collect them into a list.

        For more information about the arguments, see :meth:`iter_items`.

        :note: Favor the :meth:`iter_items` method as it will avoid eagerly collecting
            all items. When there are many items, that can slow performance and increase
            memory usage.

        :return: list(Item,...) list of item instances
        """
        out_list: IterableList = IterableList(cls._id_attribute_)
        out_list.extend(cls.iter_items(repo, *args, **kwargs))
        return out_list