from __future__ import annotations
from functools import partial, wraps
from typing import TYPE_CHECKING, TypeVar
from .. import _core
from ..abc import Clock, Instrument
Converts an async test function to be synchronous, running via Trio.

    Usage::

        @trio_test
        async def test_whatever():
            await ...

    If a pytest fixture is passed in that subclasses the :class:`~trio.abc.Clock` or
    :class:`~trio.abc.Instrument` ABCs, then those are passed to :meth:`trio.run()`.
    