import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def arg_flaglist(self, line):
    """
        Flag part of store-att-flag
        """
    flags = []
    if line[0:1] == b'(':
        if line[-1:] != b')':
            raise IllegalClientResponse('Mismatched parenthesis')
        line = line[1:-1]
    while line:
        m = self.atomre.search(line)
        if not m:
            raise IllegalClientResponse('Malformed flag')
        if line[0:1] == b'\\' and m.start() == 1:
            flags.append(b'\\' + m.group('atom'))
        elif m.start() == 0:
            flags.append(m.group('atom'))
        else:
            raise IllegalClientResponse('Malformed flag')
        line = m.group('rest')
    return (flags, b'')