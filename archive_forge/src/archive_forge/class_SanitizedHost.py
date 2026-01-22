import functools
import email.utils
import re
import builtins
from binascii import b2a_base64
from cgi import parse_header
from email.header import decode_header
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote_plus
import jaraco.collections
import cherrypy
from cherrypy._cpcompat import ntob, ntou
class SanitizedHost(str):
    """
    Wraps a raw host header received from the network in
    a sanitized version that elides dangerous characters.

    >>> SanitizedHost('foo\\nbar')
    'foobar'
    >>> SanitizedHost('foo\\nbar').raw
    'foo\\nbar'

    A SanitizedInstance is only returned if sanitization was performed.

    >>> isinstance(SanitizedHost('foobar'), SanitizedHost)
    False
    """
    dangerous = re.compile('[\\n\\r]')

    def __new__(cls, raw):
        sanitized = cls._sanitize(raw)
        if sanitized == raw:
            return raw
        instance = super().__new__(cls, sanitized)
        instance.raw = raw
        return instance

    @classmethod
    def _sanitize(cls, raw):
        return cls.dangerous.sub('', raw)