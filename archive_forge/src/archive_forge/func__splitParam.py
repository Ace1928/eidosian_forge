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
def _splitParam(cls, param):
    """
        Split an ISUPPORT parameter.

        @type param: C{str}

        @rtype: C{(str, list)}
        @return: C{(key, arguments)}
        """
    if '=' not in param:
        param += '='
    key, value = param.split('=', 1)
    return (key, [cls._unescapeParamValue(v) for v in value.split(',')])