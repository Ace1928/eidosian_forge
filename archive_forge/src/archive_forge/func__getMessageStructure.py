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
def _getMessageStructure(message):
    """
    Construct an appropriate type of message structure object for the given
    message object.

    @param message: A L{IMessagePart} provider

    @return: A L{_MessageStructure} instance of the most specific type available
        for the given message, determined by inspecting the MIME type of the
        message.
    """
    main, subtype, attrs = _getContentType(message)
    if main is not None:
        main = main.lower()
    if subtype is not None:
        subtype = subtype.lower()
    if main == 'multipart':
        return _MultipartMessageStructure(message, subtype, attrs)
    elif (main, subtype) == ('message', 'rfc822'):
        return _RFC822MessageStructure(message, main, subtype, attrs)
    elif main == 'text':
        return _TextMessageStructure(message, main, subtype, attrs)
    else:
        return _SinglepartMessageStructure(message, main, subtype, attrs)