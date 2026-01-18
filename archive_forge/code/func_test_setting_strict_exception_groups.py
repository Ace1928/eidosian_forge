from __future__ import annotations
import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
import outcome
import pytest
import sniffio
from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD
from .tutil import (
@pytest.mark.filterwarnings('ignore:.*strict_exception_groups=False:trio.TrioDeprecationWarning')
@pytest.mark.parametrize('run_strict', [True, False, None])
@pytest.mark.parametrize('open_nursery_strict', [True, False, None])
@pytest.mark.parametrize('multiple_exceptions', [True, False])
def test_setting_strict_exception_groups(run_strict: bool | None, open_nursery_strict: bool | None, multiple_exceptions: bool) -> None:
    """
    Test default values and that nurseries can both inherit and override the global context
    setting of strict_exception_groups.
    """

    async def raise_error() -> NoReturn:
        raise RuntimeError('test error')

    async def main() -> None:
        """Open a nursery, and raise one or two errors inside"""
        async with _core.open_nursery(**_create_kwargs(open_nursery_strict)) as nursery:
            nursery.start_soon(raise_error)
            if multiple_exceptions:
                nursery.start_soon(raise_error)

    def run_main() -> None:
        _core.run(main, **_create_kwargs(run_strict))
    matcher = Matcher(RuntimeError, '^test error$')
    if multiple_exceptions:
        with RaisesGroup(matcher, matcher):
            run_main()
    elif open_nursery_strict or (open_nursery_strict is None and run_strict is not False):
        with RaisesGroup(matcher):
            run_main()
    else:
        with pytest.raises(RuntimeError, match='^test error$'):
            run_main()