import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _save_keys(self, keys):
    data = {'ssh_public_keys': [{'key': '{} {}'.format(key.public_key, key.name)} for key in keys]}
    response = self.connection.request('/users/%s' % self._get_user_id(), region='account', method='PATCH', data=json.dumps(data))
    return response.success()