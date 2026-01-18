from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_zone_export_create(self):
    self.verify_create(self.proxy.create_zone_export, zone_export.ZoneExport, method_args=[{'id': 'zone_id_value'}], method_kwargs={'name': 'id'}, expected_args=[], expected_kwargs={'name': 'id', 'zone_id': 'zone_id_value', 'prepend_key': False})