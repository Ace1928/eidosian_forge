from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_recordsets(self):
    self.verify_list(self.proxy.recordsets, recordset.Recordset, expected_kwargs={'base_path': '/recordsets'})