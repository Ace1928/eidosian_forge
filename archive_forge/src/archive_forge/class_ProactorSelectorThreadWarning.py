from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
class ProactorSelectorThreadWarning(RuntimeWarning):
    """Warning class for notifying about the extra thread spawned by tornado

    We automatically support proactor via tornado's AddThreadSelectorEventLoop"""