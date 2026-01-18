from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def validate_quotas(self, project, **kwargs):
    search_opts = {'all_tenants': True, 'project_id': project}
    volume_list = None
    backup_list = None
    snapshot_list = None
    for key, value in kwargs.copy().items():
        if value == -1:
            del kwargs[key]
    if self.GIGABYTES in kwargs:
        quota_size = kwargs[self.GIGABYTES]
        volume_list = self.client().volumes.list(search_opts=search_opts)
        snapshot_list = self.client().volume_snapshots.list(search_opts=search_opts)
        total_size = sum((item.size for item in volume_list + snapshot_list))
        self._validate_quota(self.GIGABYTES, quota_size, total_size)
    if self.VOLUMES in kwargs:
        quota_size = kwargs[self.VOLUMES]
        if volume_list is None:
            volume_list = self.client().volumes.list(search_opts=search_opts)
        total_size = len(volume_list)
        self._validate_quota(self.VOLUMES, quota_size, total_size)
    if self.BACKUPS in kwargs:
        quota_size = kwargs[self.BACKUPS]
        if backup_list is None:
            backup_list = self.client().backups.list(search_opts=search_opts)
        total_size = len(backup_list)
        self._validate_quota(self.BACKUPS, quota_size, total_size)
    if self.BACKUPS_GIGABYTES in kwargs:
        quota_size = kwargs[self.BACKUPS_GIGABYTES]
        backup_list = self.client().backups.list(search_opts=search_opts)
        total_size = sum((item.size for item in backup_list))
        self._validate_quota(self.BACKUPS_GIGABYTES, quota_size, total_size)
    if self.SNAPSHOTS in kwargs:
        quota_size = kwargs[self.SNAPSHOTS]
        if snapshot_list is None:
            snapshot_list = self.client().volume_snapshots.list(search_opts=search_opts)
        total_size = len(snapshot_list)
        self._validate_quota(self.SNAPSHOTS, quota_size, total_size)