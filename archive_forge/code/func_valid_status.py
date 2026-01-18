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
def valid_status(status):
    """Return legal HTTP status Code, Reason-phrase and Message.

    The status arg must be an int, a str that begins with an int
    or the constant from ``http.client`` stdlib module.

    If status has no reason-phrase is supplied, a default reason-
    phrase will be provided.

    >>> import http.client
    >>> from http.server import BaseHTTPRequestHandler
    >>> valid_status(http.client.ACCEPTED) == (
    ...     int(http.client.ACCEPTED),
    ... ) + BaseHTTPRequestHandler.responses[http.client.ACCEPTED]
    True
    """
    if not status:
        status = 200
    code, reason = (status, None)
    if isinstance(status, str):
        code, _, reason = status.partition(' ')
        reason = reason.strip() or None
    try:
        code = int(code)
    except (TypeError, ValueError):
        raise ValueError('Illegal response status from server (%s is non-numeric).' % repr(code))
    if code < 100 or code > 599:
        raise ValueError('Illegal response status from server (%s is out of range).' % repr(code))
    if code not in response_codes:
        default_reason, message = ('', '')
    else:
        default_reason, message = response_codes[code]
    if reason is None:
        reason = default_reason
    return (code, reason, message)