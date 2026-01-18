import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType
from typing import Any, Generator, Iterator, List, Optional, Sequence, Tuple, Union
from sphinx.util import logging
from sphinx.util.inspect import isboundmethod, safe_getattr
def undecorate(subject: _MockObject) -> Any:
    """Unwrap mock if *subject* is decorated by mocked object.

    If not decorated, returns given *subject* itself.
    """
    if ismock(subject) and subject.__sphinx_decorator_args__:
        return subject.__sphinx_decorator_args__[0]
    else:
        return subject