from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePreservedStateExternalIPMapEntry(messages, interface_name, auto_delete_str='never', ip_address_literal=None, ip_address_url=None):
    return messages.PreservedState.ExternalIPsValue.AdditionalProperty(key=interface_name, value=_MakePreservedStateNetworkIP(messages, auto_delete_str, ip_address_literal, ip_address_url))