from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateBackupVault(self, backupvault_ref, async_, backup_vault):
    """Create a Cloud NetApp Backup Vault."""
    request = self.messages.NetappProjectsLocationsBackupVaultsCreateRequest(parent=backupvault_ref.Parent().RelativeName(), backupVaultId=backupvault_ref.Name(), backupVault=backup_vault)
    create_op = self.client.projects_locations_backupVaults.Create(request)
    if async_:
        return create_op
    operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)