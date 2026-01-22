from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
class DrainIpsNotSupportedError(core_exceptions.Error):
    """Raised when drain IPs are specified for Private NAT."""

    def __init__(self):
        msg = '--source-nat-drain-ips is not supported for Private NAT.'
        super(DrainIpsNotSupportedError, self).__init__(msg)