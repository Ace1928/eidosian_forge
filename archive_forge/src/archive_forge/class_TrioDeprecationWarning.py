from __future__ import annotations
import sys
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, TypeVar
import attrs
class TrioDeprecationWarning(FutureWarning):
    """Warning emitted if you use deprecated Trio functionality.

    As a young project, Trio is currently quite aggressive about deprecating
    and/or removing functionality that we realize was a bad idea. If you use
    Trio, you should subscribe to `issue #1
    <https://github.com/python-trio/trio/issues/1>`__ to get information about
    upcoming deprecations and other backwards compatibility breaking changes.

    Despite the name, this class currently inherits from
    :class:`FutureWarning`, not :class:`DeprecationWarning`, because while
    we're in young-and-aggressive mode we want these warnings to be visible by
    default. You can hide them by installing a filter or with the ``-W``
    switch: see the :mod:`warnings` documentation for details.

    """