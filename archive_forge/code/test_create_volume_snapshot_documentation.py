from openstack.block_storage.v3 import snapshot
from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base

        Test that a error status while waiting for the volume snapshot to
        create raises an exception in create_volume_snapshot.
        