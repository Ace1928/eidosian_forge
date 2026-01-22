from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class HeaderParsingError(HTTPError):
    """Raised by assert_header_parsing, but we convert it to a log.warning statement."""

    def __init__(self, defects, unparsed_data):
        message = '%s, unparsed data: %r' % (defects or 'Unknown', unparsed_data)
        super(HeaderParsingError, self).__init__(message)