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
def update_type_encryption(self, encryption=None, volume_type=None, **attrs):
    """Update a type

        :param encryption: The value can be None or a
            :class:`~openstack.block_storage.v3.type.TypeEncryption`
            instance. If this is ``None`` then ``volume_type_id`` must be
            specified.
        :param volume_type: The value can be the ID of a type or a
            :class:`~openstack.block_storage.v3.type.Type` instance.
            Required if ``encryption_id`` is None.
        :param dict attrs: The attributes to update on the type encryption.

        :returns: The updated type encryption
        :rtype: :class:`~openstack.block_storage.v3.type.TypeEncryption`
        """
    if volume_type:
        volume_type = self._get_resource(_type.Type, volume_type)
        encryption = self._get(_type.TypeEncryption, volume_type_id=volume_type.id, requires_id=False)
    return self._update(_type.TypeEncryption, encryption, **attrs)