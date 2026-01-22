from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class ActiveIpsRequiredError(core_exceptions.Error):
    """Raised when active ranges are not specified for Public NAT."""

    def __init__(self):
        msg = '--source-nat-active-ips is required for Public NAT.'
        super(ActiveIpsRequiredError, self).__init__(msg)