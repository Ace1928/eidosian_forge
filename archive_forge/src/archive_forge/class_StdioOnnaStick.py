import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
class StdioOnnaStick:
    """
    Class that pretends to be stdout/err, and turns writes into log messages.

    @ivar isError: boolean indicating whether this is stderr, in which cases
                   log messages will be logged as errors.

    @ivar encoding: unicode encoding used to encode any unicode strings
                    written to this object.
    """
    closed = 0
    softspace = 0
    mode = 'wb'
    name = '<stdio (log)>'

    def __init__(self, isError=0, encoding=None):
        self.isError = isError
        if encoding is None:
            encoding = sys.getdefaultencoding()
        self.encoding = encoding
        self.buf = ''

    def close(self):
        pass

    def fileno(self):
        return -1

    def flush(self):
        pass

    def read(self):
        raise OSError("can't read from the log!")
    readline = read
    readlines = read
    seek = read
    tell = read

    def write(self, data):
        d = (self.buf + data).split('\n')
        self.buf = d[-1]
        messages = d[0:-1]
        for message in messages:
            msg(message, printed=1, isError=self.isError)

    def writelines(self, lines):
        for line in lines:
            msg(line, printed=1, isError=self.isError)