import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _combine_cookie_fields(headers, hdr_validation_flags):
    """
    RFC 7540 § 8.1.2.5 allows HTTP/2 clients to split the Cookie header field,
    which must normally appear only once, into multiple fields for better
    compression. However, they MUST be joined back up again when received.
    This normalization step applies that transform. The side-effect is that
    all cookie fields now appear *last* in the header block.
    """
    cookies = []
    for header in headers:
        if header[0] == b'cookie':
            cookies.append(header[1])
        else:
            yield header
    if cookies:
        cookie_val = b'; '.join(cookies)
        yield NeverIndexedHeaderTuple(b'cookie', cookie_val)