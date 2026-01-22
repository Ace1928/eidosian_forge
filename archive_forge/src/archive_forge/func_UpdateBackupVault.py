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
def UpdateBackupVault(self, backupvault_ref, backup_vault, update_mask):
    """Send a Patch request for the Cloud NetApp Backup Vault."""
    update_request = self.messages.NetappProjectsLocationsBackupVaultsPatchRequest(backupVault=backup_vault, name=backupvault_ref.RelativeName(), updateMask=update_mask)
    return self.client.projects_locations_backupVaults.Patch(update_request)