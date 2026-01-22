import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _secure_headers(headers, hdr_validation_flags):
    """
    Certain headers are at risk of being attacked during the header compression
    phase, and so need to be kept out of header compression contexts. This
    function automatically transforms certain specific headers into HPACK
    never-indexed fields to ensure they don't get added to header compression
    contexts.

    This function currently implements two rules:

    - 'authorization' and 'proxy-authorization' fields are automatically made
      never-indexed.
    - Any 'cookie' header field shorter than 20 bytes long is made
      never-indexed.

    These fields are the most at-risk. These rules are inspired by Firefox
    and nghttp2.
    """
    for header in headers:
        if header[0] in _SECURE_HEADERS:
            yield NeverIndexedHeaderTuple(*header)
        elif header[0] in (b'cookie', u'cookie') and len(header[1]) < 20:
            yield NeverIndexedHeaderTuple(*header)
        else:
            yield header