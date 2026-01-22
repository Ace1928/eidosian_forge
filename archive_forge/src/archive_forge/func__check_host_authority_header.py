import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _check_host_authority_header(headers, hdr_validation_flags):
    """
    Raises a ProtocolError if a header block arrives that does not contain an
    :authority or a Host header, or if a header block contains both fields,
    but their values do not match.
    """
    skip_validation = hdr_validation_flags.is_response_header or hdr_validation_flags.is_trailer
    if skip_validation:
        return headers
    return _validate_host_authority_header(headers)