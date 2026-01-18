from __future__ import annotations
import contextlib
import heapq
import logging
import selectors
import time
import typing
from contextlib import suppress
from itertools import count
from .abstract_loop import EventLoop, ExitMainLoop

        A single iteration of the event loop
        