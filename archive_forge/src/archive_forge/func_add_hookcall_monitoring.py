from __future__ import annotations
import inspect
import types
import warnings
from typing import Any
from typing import Callable
from typing import cast
from typing import Final
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING
from . import _tracing
from ._callers import _multicall
from ._hooks import _HookImplFunction
from ._hooks import _Namespace
from ._hooks import _Plugin
from ._hooks import _SubsetHookCaller
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplOpts
from ._hooks import HookRelay
from ._hooks import HookspecOpts
from ._hooks import normalize_hookimpl_opts
from ._result import Result
def add_hookcall_monitoring(self, before: _BeforeTrace, after: _AfterTrace) -> Callable[[], None]:
    """Add before/after tracing functions for all hooks.

        Returns an undo function which, when called, removes the added tracers.

        ``before(hook_name, hook_impls, kwargs)`` will be called ahead
        of all hook calls and receive a hookcaller instance, a list
        of HookImpl instances and the keyword arguments for the hook call.

        ``after(outcome, hook_name, hook_impls, kwargs)`` receives the
        same arguments as ``before`` but also a :class:`~pluggy.Result` object
        which represents the result of the overall hook call.
        """
    oldcall = self._inner_hookexec

    def traced_hookexec(hook_name: str, hook_impls: Sequence[HookImpl], caller_kwargs: Mapping[str, object], firstresult: bool) -> object | list[object]:
        before(hook_name, hook_impls, caller_kwargs)
        outcome = Result.from_call(lambda: oldcall(hook_name, hook_impls, caller_kwargs, firstresult))
        after(outcome, hook_name, hook_impls, caller_kwargs)
        return outcome.get_result()
    self._inner_hookexec = traced_hookexec

    def undo() -> None:
        self._inner_hookexec = oldcall
    return undo