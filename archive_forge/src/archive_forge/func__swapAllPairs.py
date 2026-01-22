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
def _swapAllPairs(of, that, ifIs):
    """
    Swap each element in each pair in C{of} with C{that} it is
    C{ifIs}.

    @param of: A list of 2-L{tuple}s, whose members may be the object
        C{that}
    @type of: L{list} of 2-L{tuple}s

    @param ifIs: An object whose identity will be compared to members
        of each pair in C{of}

    @return: A L{list} of 2-L{tuple}s with all occurences of C{ifIs}
        replaced with C{that}
    """
    return [(_swap(first, that, ifIs), _swap(second, that, ifIs)) for first, second in of]