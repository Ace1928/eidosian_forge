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
class _HeadersPlusNLines:
    """
    A utility class to retrieve the header and some lines of the body of a mail
    message.

    @ivar _file: See L{__init__}
    @ivar _extraLines: See L{__init__}

    @type linecount: L{int}
    @ivar linecount: The number of full lines of the message body scanned.

    @type headers: L{bool}
    @ivar headers: An indication of which part of the message is being scanned.
        C{True} for the header and C{False} for the body.

    @type done: L{bool}
    @ivar done: A flag indicating when the desired part of the message has been
        scanned.

    @type buf: L{bytes}
    @ivar buf: The portion of the message body that has been scanned, up to
        C{n} lines.
    """

    def __init__(self, file, extraLines):
        """
        @type file: file-like object
        @param file: A file containing a mail message.

        @type extraLines: L{int}
        @param extraLines: The number of lines of the message body to retrieve.
        """
        self._file = file
        self._extraLines = extraLines
        self.linecount = 0
        self.headers = 1
        self.done = 0
        self.buf = b''

    def read(self, bytes):
        """
        Scan bytes from the file.

        @type bytes: L{int}
        @param bytes: The number of bytes to read from the file.

        @rtype: L{bytes}
        @return: Each portion of the header as it is scanned.  Then, full lines
            of the message body as they are scanned.  When more than one line
            of the header and/or body has been scanned, the result is the
            concatenation of the lines.  When the scan results in no full
            lines, the empty string is returned.
        """
        if self.done:
            return b''
        data = self._file.read(bytes)
        if not data:
            return data
        if self.headers:
            df, sz = (data.find(b'\r\n\r\n'), 4)
            if df == -1:
                df, sz = (data.find(b'\n\n'), 2)
            if df != -1:
                df += sz
                val = data[:df]
                data = data[df:]
                self.linecount = 1
                self.headers = 0
        else:
            val = b''
        if self.linecount > 0:
            dsplit = (self.buf + data).split(b'\n')
            self.buf = dsplit[-1]
            for ln in dsplit[:-1]:
                if self.linecount > self._extraLines:
                    self.done = 1
                    return val
                val += ln + b'\n'
                self.linecount += 1
            return val
        else:
            return data