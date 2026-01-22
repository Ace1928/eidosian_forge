from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class IapTunnelingUnavailable(exceptions.Error):
    """Error when IAP tunneling is unavailable (either temporarily or not)."""

    def __init__(self):
        super(IapTunnelingUnavailable, self).__init__('Currently unable to connect to this TPU using IAP tunneling.')