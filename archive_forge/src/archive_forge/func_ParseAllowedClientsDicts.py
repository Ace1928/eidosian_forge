from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def ParseAllowedClientsDicts(self, nfs_share_resource, allowed_clients_dicts):
    """Parses NFS share allowed client list of dicts."""
    allowed_clients = []
    for allowed_client in allowed_clients_dicts:
        mount_permissions = self.nfs_mount_permissions_str_to_message[allowed_client['mount-permissions']]
        network_full_name = util.NFSNetworkFullName(nfs_share_resource=nfs_share_resource, allowed_client_dict=allowed_client)
        allowed_clients.append(self.messages.AllowedClient(network=network_full_name, allowedClientsCidr=allowed_client['cidr'], mountPermissions=mount_permissions, allowDev=allowed_client['allow-dev'], allowSuid=allowed_client['allow-suid'], noRootSquash=not allowed_client['enable-root-squash']))
    return allowed_clients