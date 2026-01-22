from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.bms import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _ApplyNFSAllowedClientsUpdates(client, args, existing_nfs, nfs_share_resource):
    """Applies the changes in args to the allowed_clients in existing_nfs.

  Returns None if no changes were to be applied.

  Args:
    client: BmsClient.
    args: The arguments passed to the command.
    existing_nfs: The existing nfs.
    nfs_share_resource: The ref to the NFS share.

  Returns:
    List of allowed clients after applying updates or None if there are
    no changes.
  """
    if args.IsKnownAndSpecified('clear_allowed_clients') and args.clear_allowed_clients:
        return []
    if args.IsKnownAndSpecified('add_allowed_client'):
        new_clients = client.ParseAllowedClientsDicts(nfs_share_resource=nfs_share_resource, allowed_clients_dicts=args.add_allowed_client)
        return existing_nfs.allowedClients + new_clients
    if args.IsKnownAndSpecified('remove_allowed_client'):
        return util.RemoveAllowedClients(nfs_share_resource=nfs_share_resource, allowed_clients=existing_nfs.allowedClients, remove_key_dicts=args.remove_allowed_client)