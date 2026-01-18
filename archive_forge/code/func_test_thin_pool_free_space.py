from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_thin_pool_free_space(self):
    self.assertEqual(float('7.92'), self.vg._get_thin_pool_free_space('fake-vg', 'fake-vg-pool'))