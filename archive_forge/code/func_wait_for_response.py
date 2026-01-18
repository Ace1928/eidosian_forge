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