import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _assert_header_in_set(string_header, bytes_header, header_set):
    """
    Given a set of header names, checks whether the string or byte version of
    the header name is present. Raises a Protocol error with the appropriate
    error if it's missing.
    """
    if not (string_header in header_set or bytes_header in header_set):
        raise ProtocolError('Header block missing mandatory %s header' % string_header)