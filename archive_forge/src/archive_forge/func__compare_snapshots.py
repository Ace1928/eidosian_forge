from openstack.block_storage.v3 import snapshot
from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def _compare_snapshots(self, exp, real):
    self.assertDictEqual(snapshot.Snapshot(**exp).to_dict(computed=False), real.to_dict(computed=False))