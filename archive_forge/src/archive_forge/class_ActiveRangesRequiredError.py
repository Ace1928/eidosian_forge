from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class ActiveRangesRequiredError(core_exceptions.Error):
    """Raised when active ranges are not specified for Private NAT."""

    def __init__(self):
        msg = '--source-nat-active-ranges is required for Private NAT.'
        super(ActiveRangesRequiredError, self).__init__(msg)