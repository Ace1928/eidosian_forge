from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakePreservedStateIPAddress(messages, ip_address_literal=None, ip_address_url=None):
    """Construct a preserved state IP message."""
    if ip_address_literal is None and ip_address_url is None:
        raise ValueError('\n        For a stateful network IP you must specify either the IP or the\n        address. But the per-instance configuration specifies none.\n        ')
    elif ip_address_literal is not None and ip_address_url is not None:
        raise ValueError('\n        For a stateful network IP you must specify either the IP or the\n        address. But the per-instance configuration specifies both.\n        ')
    elif ip_address_literal is not None:
        return messages.PreservedStatePreservedNetworkIpIpAddress(literal=ip_address_literal)
    else:
        return messages.PreservedStatePreservedNetworkIpIpAddress(address=ip_address_url)