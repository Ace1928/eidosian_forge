from __future__ import annotations
import asyncio
import functools
import logging
import typing
from contextlib import suppress
from tornado import ioloop
from .abstract_loop import EventLoop, ExitMainLoop

        Remove an idle callback.

        Returns True if the handle was removed.
        