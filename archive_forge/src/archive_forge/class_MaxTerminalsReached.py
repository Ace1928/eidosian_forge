from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
class MaxTerminalsReached(Exception):
    """An error raised when we exceed the max number of terminals."""

    def __init__(self, max_terminals: int) -> None:
        """Initialize the error."""
        self.max_terminals = max_terminals

    def __str__(self) -> str:
        """The string representation of the error."""
        return 'Cannot create more than %d terminals' % self.max_terminals