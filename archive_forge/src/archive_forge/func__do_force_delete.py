from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _do_force_delete(self, snapshot, action_name='force_delete'):
    """Delete the specified snapshot ignoring its current state."""
    return self._action(action_name, base.getid(snapshot))