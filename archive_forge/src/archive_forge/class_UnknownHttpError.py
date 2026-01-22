from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class UnknownHttpError(api_exceptions.HttpException, DebugError):
    """An unknown error occurred during a remote API call."""

    def __init__(self, error):
        super(UnknownHttpError, self).__init__(error)