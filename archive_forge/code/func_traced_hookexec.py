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
def traced_hookexec(hook_name: str, hook_impls: Sequence[HookImpl], caller_kwargs: Mapping[str, object], firstresult: bool) -> object | list[object]:
    before(hook_name, hook_impls, caller_kwargs)
    outcome = Result.from_call(lambda: oldcall(hook_name, hook_impls, caller_kwargs, firstresult))
    after(outcome, hook_name, hook_impls, caller_kwargs)
    return outcome.get_result()