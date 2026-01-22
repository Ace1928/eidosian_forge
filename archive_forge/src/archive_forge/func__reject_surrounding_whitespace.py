import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _reject_surrounding_whitespace(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if any header name or value is surrounded by
    whitespace characters.
    """
    for header in headers:
        if header[0][0] in _WHITESPACE or header[0][-1] in _WHITESPACE:
            raise ProtocolError('Received header name surrounded by whitespace %r' % header[0])
        if header[1] and (header[1][0] in _WHITESPACE or header[1][-1] in _WHITESPACE):
            raise ProtocolError('Received header value surrounded by whitespace %r' % header[1])
        yield header