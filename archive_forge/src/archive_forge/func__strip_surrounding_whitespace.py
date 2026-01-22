import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _strip_surrounding_whitespace(headers, hdr_validation_flags):
    """
    Given an iterable of header two-tuples, strip both leading and trailing
    whitespace from both header names and header values. This generator
    produces tuples that preserve the original type of the header tuple for
    tuple and any ``HeaderTuple``.
    """
    for header in headers:
        if isinstance(header, HeaderTuple):
            yield header.__class__(header[0].strip(), header[1].strip())
        else:
            yield (header[0].strip(), header[1].strip())