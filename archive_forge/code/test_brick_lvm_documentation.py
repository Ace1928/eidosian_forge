from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
Test self.vg.vg_thin_pool is set to pool_name

        See bug #1220286 for more info.
        