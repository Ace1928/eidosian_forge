from oslo_utils import strutils
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
def update_snapshot_status(self, snapshot, update_dict):
    return self._action('os-update_snapshot_status', base.getid(snapshot), update_dict)