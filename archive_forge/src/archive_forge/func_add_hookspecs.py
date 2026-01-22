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
def add_hookspecs(self, module_or_class: _Namespace) -> None:
    """Add new hook specifications defined in the given ``module_or_class``.

        Functions are recognized as hook specifications if they have been
        decorated with a matching :class:`HookspecMarker`.
        """
    names = []
    for name in dir(module_or_class):
        spec_opts = self.parse_hookspec_opts(module_or_class, name)
        if spec_opts is not None:
            hc: HookCaller | None = getattr(self.hook, name, None)
            if hc is None:
                hc = HookCaller(name, self._hookexec, module_or_class, spec_opts)
                setattr(self.hook, name, hc)
            else:
                hc.set_specification(module_or_class, spec_opts)
                for hookfunction in hc.get_hookimpls():
                    self._verify_hook(hc, hookfunction)
            names.append(name)
    if not names:
        raise ValueError(f'did not find any {self.project_name!r} hooks in {module_or_class!r}')