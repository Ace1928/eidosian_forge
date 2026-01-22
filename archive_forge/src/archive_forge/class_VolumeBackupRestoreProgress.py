class VolumeBackupRestoreProgress(object):

    def __init__(self, vol_id, backup_id):
        self.called = False
        self.complete = False
        self.vol_id = vol_id
        self.backup_id = backup_id