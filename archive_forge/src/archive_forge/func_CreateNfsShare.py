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
def CreateNfsShare(self, nfs_share_resource, size_gib, storage_type, allowed_clients_dicts, labels):
    """Create an NFS share resource."""
    allowed_clients = self.ParseAllowedClientsDicts(nfs_share_resource=nfs_share_resource, allowed_clients_dicts=allowed_clients_dicts)
    nfs_share_msg = self.messages.NfsShare(name=nfs_share_resource.RelativeName(), requestedSizeGib=size_gib, storageType=self.nfs_storage_type_str_to_message[storage_type], allowedClients=allowed_clients, labels=labels)
    request = self.messages.BaremetalsolutionProjectsLocationsNfsSharesCreateRequest(nfsShare=nfs_share_msg, parent=nfs_share_resource.Parent().RelativeName())
    return self.nfs_shares_service.Create(request)