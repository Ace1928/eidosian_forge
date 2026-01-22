import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _lowercase_header_names(headers, hdr_validation_flags):
    """
    Given an iterable of header two-tuples, rebuilds that iterable with the
    header names lowercased. This generator produces tuples that preserve the
    original type of the header tuple for tuple and any ``HeaderTuple``.
    """
    for header in headers:
        if isinstance(header, HeaderTuple):
            yield header.__class__(header[0].lower(), header[1])
        else:
            yield (header[0].lower(), header[1])