from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnknownApiEndpointOverrideError(exceptions.Error):
    """An error raised for an invalid value for `api_endpoint_overrides`."""

    def __init__(self, api_name):
        message = 'Unknown api_endpoint_overrides value for {}'.format(api_name)
        super(UnknownApiEndpointOverrideError, self).__init__(message)