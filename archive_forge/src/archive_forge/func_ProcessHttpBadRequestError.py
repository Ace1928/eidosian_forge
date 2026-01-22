from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def ProcessHttpBadRequestError(error):
    """Intercept INVALID_ARGUMENT errors related to checksum verification.

  Intercept INVALID_ARGUMENT errors related to checksum verification, to present
  a user-friendly message.
  All other errors are surfaced as-is.
  Args:
    error: apitools_exceptions.ProcessHttpBadRequestError.

  Raises:
    ServerSideIntegrityVerificationError: if |error| is a result of a failed
    server-side request integrity verification.
    Else, re-raises |error| as exceptions.HttpException.
  """
    exc = exceptions.HttpException(error)
    regex = re.compile('The checksum in field .* did not match the data in field .*.')
    if regex.search(exc.payload.status_message) is not None:
        raise ServerSideIntegrityVerificationError(GetRequestToServerCorruptedErrorMessage())
    else:
        raise exc