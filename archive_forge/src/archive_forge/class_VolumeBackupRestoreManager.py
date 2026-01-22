from cinderclient import base
class VolumeBackupRestoreManager(base.Manager):
    """Manage :class:`VolumeBackupsRestore` resources."""
    resource_class = VolumeBackupsRestore

    def restore(self, backup_id, volume_id=None, name=None):
        """Restore a backup to a volume.

        :param backup_id: The ID of the backup to restore.
        :param volume_id: The ID of the volume to restore the backup to.
        :param name     : The name for new volume creation to restore.
        :rtype: :class:`Restore`
        """
        body = {'restore': {'volume_id': volume_id, 'name': name}}
        return self._create('/backups/%s/restore' % backup_id, body, 'restore')