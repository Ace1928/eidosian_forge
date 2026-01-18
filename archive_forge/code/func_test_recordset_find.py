from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_recordset_find(self):
    self._verify('openstack.proxy.Proxy._find', self.proxy.find_recordset, method_args=['zone', 'rs'], method_kwargs={}, expected_args=[recordset.Recordset, 'rs'], expected_kwargs={'ignore_missing': True, 'zone_id': 'zone'})