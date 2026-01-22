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
def _extraInfo(self, lines):
    flags = {}
    recent = exists = None
    for response in lines:
        elements = len(response)
        if elements == 1 and response[0] == [b'READ-ONLY']:
            self.modeChanged(False)
        elif elements == 1 and response[0] == [b'READ-WRITE']:
            self.modeChanged(True)
        elif elements == 2 and response[1] == b'EXISTS':
            exists = int(response[0])
        elif elements == 2 and response[1] == b'RECENT':
            recent = int(response[0])
        elif elements == 3 and response[1] == b'FETCH':
            mId = int(response[0])
            values, _ = self._parseFetchPairs(response[2])
            flags.setdefault(mId, []).extend(values.get('FLAGS', ()))
        else:
            log.msg(f'Unhandled unsolicited response: {response}')
    if flags:
        self.flagsChanged(flags)
    if recent is not None or exists is not None:
        self.newMessages(exists, recent)