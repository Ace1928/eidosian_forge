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
def _listWork(self, tag, ref, mbox, sub, cmdName):
    mbox = _parseMbox(mbox)
    ref = _parseMbox(ref)
    maybeDeferred(self.account.listMailboxes, ref, mbox).addCallback(self._cbListWork, tag, sub, cmdName).addErrback(self._ebListWork, tag)