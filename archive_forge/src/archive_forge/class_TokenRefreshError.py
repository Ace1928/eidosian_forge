from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class TokenRefreshError(AuthenticationException):
    """An exception raised when the auth tokens fail to refresh."""

    def __init__(self, error, for_adc=False, should_relogin=True):
        message = 'There was a problem refreshing your current auth tokens: {0}'.format(error)
        super(TokenRefreshError, self).__init__(message, for_adc=for_adc, should_relogin=should_relogin)