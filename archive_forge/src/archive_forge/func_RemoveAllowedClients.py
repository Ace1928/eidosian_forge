from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def RemoveAllowedClients(nfs_share_resource, allowed_clients, remove_key_dicts):
    """Removes the allowed clients specified by remove_key_dicts from allowed_clients."""
    keys_to_remove = set()
    for key_dict in remove_key_dicts:
        key_network_full_name = NFSNetworkFullName(nfs_share_resource=nfs_share_resource, allowed_client_dict=key_dict)
        keys_to_remove.add((key_network_full_name, key_dict['cidr']))
    out = []
    for allowed_client in allowed_clients:
        curr_key = (allowed_client.network, allowed_client.allowedClientsCidr)
        if curr_key in keys_to_remove:
            keys_to_remove.remove(curr_key)
        else:
            out.append(allowed_client)
    for key in keys_to_remove:
        raise LookupError('Cannot find an existing allowed client for network [{}] and CIDR [{}]'.format(key[0], key[1]))
    return out