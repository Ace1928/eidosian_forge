from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _do_access_list(self, snapshot):
    snapshot_id = base.getid(snapshot)
    access_list = self._list('/snapshots/%s/access-list' % snapshot_id, 'snapshot_access_list')
    return access_list