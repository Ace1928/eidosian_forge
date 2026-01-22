from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class ActiveRangesNotSupportedError(core_exceptions.Error):
    """Raised when active ranges are specified for Public NAT."""

    def __init__(self):
        msg = '--source-nat-active-ranges is not supported for Public NAT.'
        super(ActiveRangesNotSupportedError, self).__init__(msg)