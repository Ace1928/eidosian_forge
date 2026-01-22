from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
class OutgoingRequest(Request):
    """Represents an outgoing request, for which it is possible to wait for a
    response to be received, and register a response handler.
    """
    _parse = _handle = None

    def __init__(self, channel, seq, command, arguments):
        super().__init__(channel, seq, command, arguments)
        self._response_handlers = []

    def describe(self):
        return f'{self.seq} request {json.repr(self.command)} to {self.channel}'

    def wait_for_response(self, raise_if_failed=True):
        """Waits until a response is received for this request, records the Response
        object for it in self.response, and returns response.body.

        If no response was received from the other party before the channel closed,
        self.response is a synthesized Response with body=NoMoreMessages().

        If raise_if_failed=True and response.success is False, raises response.body
        instead of returning.
        """
        with self.channel:
            while self.response is None:
                self.channel._handlers_enqueued.wait()
        if raise_if_failed and (not self.response.success):
            raise self.response.body
        return self.response.body

    def on_response(self, response_handler):
        """Registers a handler to invoke when a response is received for this request.
        The handler is invoked with Response as its sole argument.

        If response has already been received, invokes the handler immediately.

        It is guaranteed that self.response is set before the handler is invoked.
        If no response was received from the other party before the channel closed,
        self.response is a dummy Response with body=NoMoreMessages().

        The handler is always invoked asynchronously on an unspecified background
        thread - thus, the caller of on_response() can never be blocked or deadlocked
        by the handler.

        No further incoming messages are processed until the handler returns, except for
        responses to requests that have wait_for_response() invoked on them.
        """
        with self.channel:
            self._response_handlers.append(response_handler)
            self._enqueue_response_handlers()

    def _enqueue_response_handlers(self):
        response = self.response
        if response is None:
            return

        def run_handlers():
            for handler in handlers:
                try:
                    try:
                        handler(response)
                    except MessageHandlingError as exc:
                        if not exc.applies_to(response):
                            raise
                        log.error("Handler {0}\ncouldn't handle {1}:\n{2}", util.srcnameof(handler), response.describe(), str(exc))
                except Exception:
                    log.reraise_exception("Handler {0}\ncouldn't handle {1}:", util.srcnameof(handler), response.describe())
        handlers = self._response_handlers[:]
        self.channel._enqueue_handlers(response, run_handlers)
        del self._response_handlers[:]