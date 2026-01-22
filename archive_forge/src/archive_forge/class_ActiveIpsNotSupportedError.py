from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class ActiveIpsNotSupportedError(core_exceptions.Error):
    """Raised when active IPs are specified for Private NAT."""

    def __init__(self):
        msg = '--source-nat-active-ips is not supported for Private NAT.'
        super(ActiveIpsNotSupportedError, self).__init__(msg)