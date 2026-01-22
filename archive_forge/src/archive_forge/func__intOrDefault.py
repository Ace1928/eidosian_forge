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
def _intOrDefault(value, default=None):
    """
    Convert a value to an integer if possible.

    @rtype: C{int} or type of L{default}
    @return: An integer when C{value} can be converted to an integer,
        otherwise return C{default}
    """
    if value:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return default