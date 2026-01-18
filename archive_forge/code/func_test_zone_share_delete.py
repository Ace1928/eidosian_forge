from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_zone_share_delete(self):
    self.verify_delete(self.proxy.delete_zone_share, zone_share.ZoneShare, ignore_missing=True, method_args={'zone': 'bogus_id', 'zone_share': 'bogus_id'}, expected_args=['zone_share'], expected_kwargs={'zone_id': 'zone', 'ignore_missing': True})