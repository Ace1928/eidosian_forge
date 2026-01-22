from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _do_allow(self, snapshot, access_type, access_to):
    access_params = {'access_type': access_type, 'access_to': access_to}
    return self._action('allow_access', snapshot, access_params)[1]['snapshot_access']