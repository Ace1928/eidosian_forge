from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetSubnetMode(network):
    """Returns the subnet mode of the input network."""
    if network.get('IPv4Range') is not None:
        return 'LEGACY'
    elif network.get('autoCreateSubnetworks'):
        return 'AUTO'
    else:
        return 'CUSTOM'