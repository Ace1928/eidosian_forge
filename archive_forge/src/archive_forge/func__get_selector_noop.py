from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
def _get_selector_noop(loop) -> asyncio.AbstractEventLoop:
    """no-op on non-Windows"""
    return loop