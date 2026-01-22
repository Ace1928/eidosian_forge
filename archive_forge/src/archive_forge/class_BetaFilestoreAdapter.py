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
class BetaFilestoreAdapter(AlphaFilestoreAdapter):
    """Adapter for the beta filestore API."""

    def __init__(self):
        super(BetaFilestoreAdapter, self).__init__()
        self.client = GetClient(version=BETA_API_VERSION)
        self.messages = GetMessages(version=BETA_API_VERSION)

    def _ParseSourceBackupFromFileshare(self, file_share):
        if 'source-backup' not in file_share:
            return None
        project = properties.VALUES.core.project.Get(required=True)
        location = file_share.get('source-backup-region')
        if location is None:
            raise InvalidArgumentError("If 'source-backup' is specified, 'source-backup-region' must also be specified.")
        return backup_util.BACKUP_NAME_TEMPLATE.format(project, location, file_share.get('source-backup'))

    def ParseManagedADIntoInstance(self, instance, managed_ad):
        """Parses managed-ad configs into an instance message.

    Args:
      instance: The filestore instance struct.
      managed_ad: The managed_ad cli paramters

    Raises:
      InvalidArgumentError: If managed_ad argument constraints are violated.
    """
        domain = managed_ad.get('domain')
        if domain is None:
            raise InvalidArgumentError('Domain parameter is missing in --managed_ad.')
        computer = managed_ad.get('computer')
        if computer is None:
            raise InvalidArgumentError('Computer parameter is missing in --managed_ad.')
        instance.directoryServices = self.messages.DirectoryServicesConfig(managedActiveDirectory=self.messages.ManagedActiveDirectoryConfig(domain=domain, computer=computer))

    def ParseFileShareIntoInstance(self, instance, file_share, instance_zone=None):
        """Parse specified file share configs into an instance message."""
        del instance_zone
        if instance.fileShares is None:
            instance.fileShares = []
        if file_share:
            source_backup = None
            location = None
            instance.fileShares = [fs for fs in instance.fileShares if fs.name != file_share.get('name')]
            if 'source-backup' in file_share:
                _ = properties.VALUES.core.project.Get(required=True)
                location = file_share.get('source-backup-region')
                if location is None:
                    raise InvalidArgumentError("If 'source-backup' is specified, 'source-backup-region' must also be specified.")
            source_backup = self._ParseSourceBackupFromFileshare(file_share)
            nfs_export_options = FilestoreClient.MakeNFSExportOptionsMsgBeta(self.messages, file_share.get('nfs-export-options', []))
            file_share_config = self.messages.FileShareConfig(name=file_share.get('name'), capacityGb=utils.BytesToGb(file_share.get('capacity')), sourceBackup=source_backup, nfsExportOptions=nfs_export_options)
            instance.fileShares.append(file_share_config)

    def FileSharesFromInstance(self, instance):
        """Get fileshare configs from instance message."""
        return instance.fileShares