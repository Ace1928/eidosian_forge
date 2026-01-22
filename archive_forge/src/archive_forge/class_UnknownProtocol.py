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
class UnknownProtocol(HTTPException):

    def __init__(self, version):
        self.args = (version,)
        self.version = version