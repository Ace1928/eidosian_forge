import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
class TestLoop(base_events.BaseEventLoop):
    """Loop for unittests.

    It manages self time directly.
    If something scheduled to be executed later then
    on next loop iteration after all ready handlers done
    generator passed to __init__ is calling.

    Generator should be like this:

        def gen():
            ...
            when = yield ...
            ... = yield time_advance

    Value returned by yield is absolute time of next scheduled handler.
    Value passed to yield is time advance to move loop's time forward.
    """

    def __init__(self, gen=None):
        super().__init__()
        if gen is None:

            def gen():
                yield
            self._check_on_close = False
        else:
            self._check_on_close = True
        self._gen = gen()
        next(self._gen)
        self._time = 0
        self._clock_resolution = 1e-09
        self._timers = []
        self._selector = TestSelector()
        self.readers = {}
        self.writers = {}
        self.reset_counters()

    def time(self):
        return self._time

    def advance_time(self, advance):
        """Move test time forward."""
        if advance:
            self._time += advance

    def close(self):
        super().close()
        if self._check_on_close:
            try:
                self._gen.send(0)
            except StopIteration:
                pass
            else:
                raise AssertionError('Time generator is not finished')

    def add_reader(self, fd, callback, *args):
        self.readers[fd] = events.Handle(callback, args, self)

    def remove_reader(self, fd):
        self.remove_reader_count[fd] += 1
        if fd in self.readers:
            del self.readers[fd]
            return True
        else:
            return False

    def assert_reader(self, fd, callback, *args):
        assert fd in self.readers, 'fd {} is not registered'.format(fd)
        handle = self.readers[fd]
        assert handle._callback == callback, '{!r} != {!r}'.format(handle._callback, callback)
        assert handle._args == args, '{!r} != {!r}'.format(handle._args, args)

    def add_writer(self, fd, callback, *args):
        self.writers[fd] = events.Handle(callback, args, self)

    def remove_writer(self, fd):
        self.remove_writer_count[fd] += 1
        if fd in self.writers:
            del self.writers[fd]
            return True
        else:
            return False

    def assert_writer(self, fd, callback, *args):
        assert fd in self.writers, 'fd {} is not registered'.format(fd)
        handle = self.writers[fd]
        assert handle._callback == callback, '{!r} != {!r}'.format(handle._callback, callback)
        assert handle._args == args, '{!r} != {!r}'.format(handle._args, args)

    def reset_counters(self):
        self.remove_reader_count = collections.defaultdict(int)
        self.remove_writer_count = collections.defaultdict(int)

    def _run_once(self):
        super()._run_once()
        for when in self._timers:
            advance = self._gen.send(when)
            self.advance_time(advance)
        self._timers = []

    def call_at(self, when, callback, *args):
        self._timers.append(when)
        return super().call_at(when, callback, *args)

    def _process_events(self, event_list):
        return

    def _write_to_self(self):
        pass