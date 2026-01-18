from cinderclient import base
Restore a backup to a volume.

        :param backup_id: The ID of the backup to restore.
        :param volume_id: The ID of the volume to restore the backup to.
        :param name     : The name for new volume creation to restore.
        :rtype: :class:`Restore`
        