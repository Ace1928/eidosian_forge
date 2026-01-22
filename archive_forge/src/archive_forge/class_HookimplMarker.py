from __future__ import annotations
import inspect
import sys
import warnings
from types import ModuleType
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Final
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from ._result import Result
@final
class HookimplMarker:
    """Decorator for marking functions as hook implementations.

    Instantiate it with a ``project_name`` to get a decorator.
    Calling :meth:`PluginManager.register` later will discover all marked
    functions if the :class:`PluginManager` uses the same project name.
    """
    __slots__ = ('project_name',)

    def __init__(self, project_name: str) -> None:
        self.project_name: Final = project_name

    @overload
    def __call__(self, function: _F, hookwrapper: bool=..., optionalhook: bool=..., tryfirst: bool=..., trylast: bool=..., specname: str | None=..., wrapper: bool=...) -> _F:
        ...

    @overload
    def __call__(self, function: None=..., hookwrapper: bool=..., optionalhook: bool=..., tryfirst: bool=..., trylast: bool=..., specname: str | None=..., wrapper: bool=...) -> Callable[[_F], _F]:
        ...

    def __call__(self, function: _F | None=None, hookwrapper: bool=False, optionalhook: bool=False, tryfirst: bool=False, trylast: bool=False, specname: str | None=None, wrapper: bool=False) -> _F | Callable[[_F], _F]:
        """If passed a function, directly sets attributes on the function
        which will make it discoverable to :meth:`PluginManager.register`.

        If passed no function, returns a decorator which can be applied to a
        function later using the attributes supplied.

        :param optionalhook:
            If ``True``, a missing matching hook specification will not result
            in an error (by default it is an error if no matching spec is
            found). See :ref:`optionalhook`.

        :param tryfirst:
            If ``True``, this hook implementation will run as early as possible
            in the chain of N hook implementations for a specification. See
            :ref:`callorder`.

        :param trylast:
            If ``True``, this hook implementation will run as late as possible
            in the chain of N hook implementations for a specification. See
            :ref:`callorder`.

        :param wrapper:
            If ``True`` ("new-style hook wrapper"), the hook implementation
            needs to execute exactly one ``yield``. The code before the
            ``yield`` is run early before any non-hook-wrapper function is run.
            The code after the ``yield`` is run after all non-hook-wrapper
            functions have run. The ``yield`` receives the result value of the
            inner calls, or raises the exception of inner calls (including
            earlier hook wrapper calls). The return value of the function
            becomes the return value of the hook, and a raised exception becomes
            the exception of the hook. See :ref:`hookwrapper`.

        :param hookwrapper:
            If ``True`` ("old-style hook wrapper"), the hook implementation
            needs to execute exactly one ``yield``. The code before the
            ``yield`` is run early before any non-hook-wrapper function is run.
            The code after the ``yield`` is run after all non-hook-wrapper
            function have run  The ``yield`` receives a :class:`Result` object
            representing the exception or result outcome of the inner calls
            (including earlier hook wrapper calls). This option is mutually
            exclusive with ``wrapper``. See :ref:`old_style_hookwrapper`.

        :param specname:
            If provided, the given name will be used instead of the function
            name when matching this hook implementation to a hook specification
            during registration. See :ref:`specname`.

        .. versionadded:: 1.2.0
            The ``wrapper`` parameter.
        """

        def setattr_hookimpl_opts(func: _F) -> _F:
            opts: HookimplOpts = {'wrapper': wrapper, 'hookwrapper': hookwrapper, 'optionalhook': optionalhook, 'tryfirst': tryfirst, 'trylast': trylast, 'specname': specname}
            setattr(func, self.project_name + '_impl', opts)
            return func
        if function is None:
            return setattr_hookimpl_opts
        else:
            return setattr_hookimpl_opts(function)