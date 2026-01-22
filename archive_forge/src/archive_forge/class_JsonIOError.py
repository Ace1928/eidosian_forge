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
class JsonIOError(IOError):
    """Indicates that a read or write operation on JsonIOStream has failed."""

    def __init__(self, *args, **kwargs):
        stream = kwargs.pop('stream')
        cause = kwargs.pop('cause', None)
        if not len(args) and cause is not None:
            args = [str(cause)]
        super().__init__(*args, **kwargs)
        self.stream = stream
        "The stream that couldn't be read or written.\n\n        Set by JsonIOStream.read_json() and JsonIOStream.write_json().\n\n        JsonMessageChannel relies on this value to decide whether a NoMoreMessages\n        instance that bubbles up to the message loop is related to that loop.\n        "
        self.cause = cause
        'The underlying exception, if any.'