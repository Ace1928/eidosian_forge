import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def _live_migrate(self, session, host, force, block_migration, disk_over_commit):
    microversion = None
    body: ty.Dict[str, ty.Any] = {'host': None}
    if block_migration == 'auto':
        raise ValueError("Live migration on this cloud does not support 'auto' as a parameter to block_migration, but only True and False.")
    body['block_migration'] = block_migration or False
    body['disk_over_commit'] = disk_over_commit or False
    if host:
        body['host'] = host
        if not force:
            raise ValueError("Live migration on this cloud implies 'force' if the 'host' option has been given and it is not possible to disable. It is recommended to not use 'host' at all on this cloud as it is inherently unsafe, but if it is unavoidable, please supply 'force=True' so that it is clear you understand the risks.")
    self._action(session, {'os-migrateLive': body}, microversion=microversion)