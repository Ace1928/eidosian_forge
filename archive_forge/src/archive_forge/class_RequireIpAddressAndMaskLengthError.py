from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
class RequireIpAddressAndMaskLengthError(RouterError):
    """Raised when ip-address or mask-length is specified without the other."""

    def __init__(self):
        msg = _REQUIRE_IP_ADDRESS_AND_MASK_LENGTH_ERROR_MESSAGE
        super(RequireIpAddressAndMaskLengthError, self).__init__(msg)