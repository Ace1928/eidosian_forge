import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def AcceptableMimeType(accept_patterns, mime_type):
    """Return True iff mime_type is acceptable for one of accept_patterns.

    Note that this function assumes that all patterns in accept_patterns
    will be simple types of the form "type/subtype", where one or both
    of these can be "*". We do not support parameters (i.e. "; q=") in
    patterns.

    Args:
      accept_patterns: list of acceptable MIME types.
      mime_type: the mime type we would like to match.

    Returns:
      Whether or not mime_type matches (at least) one of these patterns.
    """
    if '/' not in mime_type:
        raise exceptions.InvalidUserInputError('Invalid MIME type: "%s"' % mime_type)
    unsupported_patterns = [p for p in accept_patterns if ';' in p]
    if unsupported_patterns:
        raise exceptions.GeneratedClientError('MIME patterns with parameter unsupported: "%s"' % ', '.join(unsupported_patterns))

    def MimeTypeMatches(pattern, mime_type):
        """Return True iff mime_type is acceptable for pattern."""
        if pattern == '*':
            pattern = '*/*'
        return all((accept in ('*', provided) for accept, provided in zip(pattern.split('/'), mime_type.split('/'))))
    return any((MimeTypeMatches(pattern, mime_type) for pattern in accept_patterns))