import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
@staticmethod
def _is_textIO(stream):
    """Test whether a file-like object is a text or a binary stream.
        """
    return isinstance(stream, io.TextIOBase)