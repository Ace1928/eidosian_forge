from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakePreservedStateNetworkIP(messages, auto_delete_str, ip_address_literal, ip_address_url):
    return messages.PreservedStatePreservedNetworkIp(autoDelete=_MakePreservedStateIPAutoDelete(messages, auto_delete_str), ipAddress=_MakePreservedStateIPAddress(messages, ip_address_literal, ip_address_url))