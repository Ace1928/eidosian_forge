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
def unbare_repo(func: Callable[..., T]) -> Callable[..., T]:
    """Methods with this decorator raise :class:`.exc.InvalidGitRepositoryError` if they
    encounter a bare repository."""
    from .exc import InvalidGitRepositoryError

    @wraps(func)
    def wrapper(self: 'Remote', *args: Any, **kwargs: Any) -> T:
        if self.repo.bare:
            raise InvalidGitRepositoryError("Method '%s' cannot operate on bare repositories" % func.__name__)
        return func(self, *args, **kwargs)
    return wrapper