from __future__ import annotations
import warnings
from typing import cast
from typing import Generator
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union
from ._hooks import HookImpl
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning
def _warn_teardown_exception(hook_name: str, hook_impl: HookImpl, e: BaseException) -> None:
    msg = 'A plugin raised an exception during an old-style hookwrapper teardown.\n'
    msg += f'Plugin: {hook_impl.plugin_name}, Hook: {hook_name}\n'
    msg += f'{type(e).__name__}: {e}\n'
    msg += 'For more information see https://pluggy.readthedocs.io/en/stable/api_reference.html#pluggy.PluggyTeardownRaisedWarning'
    warnings.warn(PluggyTeardownRaisedWarning(msg), stacklevel=5)