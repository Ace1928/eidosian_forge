import typing as ty
from openstack.block_storage import _base_proxy
from openstack.block_storage.v3 import attachment as _attachment
from openstack.block_storage.v3 import availability_zone
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import block_storage_summary as _summary
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import limits as _limits
from openstack.block_storage.v3 import quota_set as _quota_set
from openstack.block_storage.v3 import resource_filter as _resource_filter
from openstack.block_storage.v3 import service as _service
from openstack.block_storage.v3 import snapshot as _snapshot
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import transfer as _transfer
from openstack.block_storage.v3 import type as _type
from openstack.block_storage.v3 import volume as _volume
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack import resource
from openstack import utils
def transfers(self, *, details=True, all_projects=False, **query):
    """Retrieve a generator of transfers

        :param bool details: When set to ``False`` no extended attributes
            will be returned. The default, ``True``, will cause objects with
            additional attributes to be returned.
        :param bool all_projects: When set to ``True``, list transfers from
            all projects. Admin-only by default.
        :param kwargs query: Optional query parameters to be sent to limit
            the transfers being returned.

        :returns: A generator of transfer objects.
        """
    if all_projects:
        query['all_projects'] = True
    base_path = '/volume-transfers'
    if not utils.supports_microversion(self, '3.55'):
        base_path = '/os-volume-transfer'
    if details:
        base_path = utils.urljoin(base_path, 'detail')
    return self._list(_transfer.Transfer, base_path=base_path, **query)