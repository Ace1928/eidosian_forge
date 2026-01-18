from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_volume_create_after_thin_creation(self):
    """Test self.vg.vg_thin_pool is set to pool_name

        See bug #1220286 for more info.
        """
    vg_name = 'vg-name'
    pool_name = vg_name + '-pool'
    pool_path = '%s/%s' % (vg_name, pool_name)

    def executor(obj, *cmd, **kwargs):
        self.assertEqual(pool_path, cmd[-1])
    self.vg._executor = executor
    self.vg.create_thin_pool(pool_name)
    self.vg.create_volume('test', '1G', lv_type='thin')
    self.assertEqual(pool_name, self.vg.vg_thin_pool)