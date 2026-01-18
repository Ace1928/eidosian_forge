import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
def wait_until(fn, page=None, timeout=5000, interval=100):
    """
    Exercice a test function in a loop until it evaluates to True
    or times out.

    The function can either be a simple lambda that returns True or False:
    >>> wait_until(lambda: x.values() == ['x'])

    Or a defined function with an assert:
    >>> def _()
    >>>    assert x.values() == ['x']
    >>> wait_until(_)

    In a Playwright context test you should pass the page fixture:
    >>> wait_until(lambda: x.values() == ['x'], page)

    Parameters
    ----------
    fn : callable
        Callback
    page : playwright.sync_api.Page, optional
        Playwright page
    timeout : int, optional
        Total timeout in milliseconds, by default 5000
    interval : int, optional
        Waiting interval, by default 100

    Adapted from pytest-qt.
    """
    __tracebackhide__ = True
    start = time.time()

    def timed_out():
        elapsed = time.time() - start
        elapsed_ms = elapsed * 1000
        return elapsed_ms > timeout
    timeout_msg = f'wait_until timed out in {timeout} milliseconds'
    while True:
        try:
            result = fn()
        except AssertionError as e:
            if timed_out():
                raise TimeoutError(timeout_msg) from e
        else:
            if result not in (None, True, False):
                raise ValueError(f'`wait_until` callback must return None, True or False, returned {result!r}')
            if result is None:
                return
            if result:
                return
            if timed_out():
                raise TimeoutError(timeout_msg)
        if page:
            page.wait_for_timeout(interval)
        else:
            time.sleep(interval / 1000)