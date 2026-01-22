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
class Disconnect(Message):
    """A dummy message used to represent disconnect. It's always the last message
    received from any channel.
    """

    def __init__(self, channel):
        super().__init__(channel, None)

    def describe(self):
        return f'disconnect from {self.channel}'