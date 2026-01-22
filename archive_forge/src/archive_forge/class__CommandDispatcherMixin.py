import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class _CommandDispatcherMixin:
    """
    Dispatch commands to handlers based on their name.

    Command handler names should be of the form C{prefix_commandName},
    where C{prefix} is the value specified by L{prefix}, and must
    accept the parameters as given to L{dispatch}.

    Attempting to mix this in more than once for a single class will cause
    strange behaviour, due to L{prefix} being overwritten.

    @type prefix: C{str}
    @ivar prefix: Command handler prefix, used to locate handler attributes
    """
    prefix: Optional[str] = None

    def dispatch(self, commandName, *args):
        """
        Perform actual command dispatch.
        """

        def _getMethodName(command):
            return f'{self.prefix}_{command}'

        def _getMethod(name):
            return getattr(self, _getMethodName(name), None)
        method = _getMethod(commandName)
        if method is not None:
            return method(*args)
        method = _getMethod('unknown')
        if method is None:
            raise UnhandledCommand(f'No handler for {_getMethodName(commandName)!r} could be found')
        return method(commandName, *args)