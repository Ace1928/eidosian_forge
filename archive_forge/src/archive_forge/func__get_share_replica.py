from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _get_share_replica(self, replica):
    """Get a share replica.

        :param replica: either replica object or its UUID.
        :rtype: :class:`ShareReplica`
        """
    replica_id = base.getid(replica)
    return self._get(RESOURCE_PATH % replica_id, RESOURCE_NAME)