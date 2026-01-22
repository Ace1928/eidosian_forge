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
def _payload(value):
    """JSON validator for message payload.

    If that value is missing or null, it is treated as if it were {}.
    """
    if value is not None and value != ():
        if isinstance(value, dict):
            assert isinstance(value, MessageDict)
        return value

    def associate_with(message):
        value.message = message
    value = MessageDict(None)
    value.associate_with = associate_with
    return value