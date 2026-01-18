import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def transformChunk(self, chunk):
    """
        Transform a chunk of a message to POP3 message format.

        Make sure each line ends with C{'\\r\\n'} and byte-stuff the
        termination character (C{'.'}) by adding an extra one when one appears
        at the beginning of a line.

        @type chunk: L{bytes}
        @param chunk: A string to transform.

        @rtype: L{bytes}
        @return: The transformed string.
        """
    return chunk.replace(b'\n', b'\r\n').replace(b'\r\n.', b'\r\n..')