from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseUpdatedInstanceConfig(self, instance_config, description=None, labels=None, file_share=None, managed_ad=None, disconnect_managed_ad=None, clear_nfs_export_options=False):
    """Parse update information into an updated Instance message."""
    if description:
        instance_config.description = description
    if labels:
        instance_config.labels = labels
    if file_share:
        self.ValidateFileShareForUpdate(instance_config, file_share)
        orig_nfs_export_options = []
        if instance_config.fileShares[0] and instance_config.fileShares[0].nfsExportOptions:
            orig_nfs_export_options = instance_config.fileShares[0].nfsExportOptions
        self.ParseFileShareIntoInstance(instance_config, file_share)
        if not instance_config.fileShares[0].nfsExportOptions:
            instance_config.fileShares[0].nfsExportOptions = orig_nfs_export_options
        if clear_nfs_export_options:
            instance_config.fileShares[0].nfsExportOptions = []
    if managed_ad:
        self.ParseManagedADIntoInstance(instance_config, managed_ad)
    if disconnect_managed_ad:
        instance_config.directoryServices = None
    return instance_config