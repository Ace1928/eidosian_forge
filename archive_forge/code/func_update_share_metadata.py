from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import limit as _limit
from openstack.shared_file_system.v2 import resource_locks as _resource_locks
from openstack.shared_file_system.v2 import share as _share
from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_instance as _share_instance
from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import share_snapshot as _share_snapshot
from openstack.shared_file_system.v2 import (
from openstack.shared_file_system.v2 import storage_pool as _storage_pool
from openstack.shared_file_system.v2 import user_message as _user_message
def update_share_metadata(self, share_id, metadata, replace=False):
    """Updates metadata of given share.

        :param share_id: The ID of the share
        :param metadata: The metadata to be created
        :param replace: Boolean for whether the preexisting metadata
            should be replaced

        :returns: A :class:`~openstack.shared_file_system.v2.share.Share`
            with the share's updated metadata.
        :rtype:
            :class:`~openstack.shared_file_system.v2.share.Share`
        """
    share = self._get_resource(_share.Share, share_id)
    return share.set_metadata(self, metadata=metadata, replace=replace)