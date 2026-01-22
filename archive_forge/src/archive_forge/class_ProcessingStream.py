import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class ProcessingStream(object):
    """
        Stream-like object that takes each completed line written to it,
        adds a given prefix, and applies the given function to it.

        .. versionadded:: 2.2.0
    """

    def __init__(self, channel, func):
        self.buffer = ''
        self.func = func
        self.channel = channel
        self.errors = ''

    def write(self, s):
        s = self.buffer + s
        self.flush()
        f = self.func
        channel = self.channel
        lines = s.split('\n')
        for line in lines[:-1]:
            f('%s: %s' % (channel, line))
        self.buffer = lines[-1]

    def flush(self):
        return

    def isatty(self):
        return False