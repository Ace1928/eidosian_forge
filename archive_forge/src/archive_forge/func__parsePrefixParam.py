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
@classmethod
def _parsePrefixParam(cls, prefix):
    """
        Parse the ISUPPORT "PREFIX" parameter.

        The order in which the parameter arguments appear is significant, the
        earlier a mode appears the more privileges it gives.

        @rtype: C{dict} mapping C{str} to C{(str, int)}
        @return: A dictionary mapping a mode character to a two-tuple of
            C({symbol, priority)}, the lower a priority (the lowest being
            C{0}) the more privileges it gives
        """
    if not prefix:
        return None
    if prefix[0] != '(' and ')' not in prefix:
        raise ValueError('Malformed PREFIX parameter')
    modes, symbols = prefix.split(')', 1)
    symbols = zip(symbols, range(len(symbols)))
    modes = modes[1:]
    return dict(zip(modes, symbols))