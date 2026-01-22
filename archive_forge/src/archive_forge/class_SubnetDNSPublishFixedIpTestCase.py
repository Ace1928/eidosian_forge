from neutron_lib.api.definitions import subnet_dns_publish_fixed_ip as sn
from neutron_lib.tests.unit.api.definitions import base
class SubnetDNSPublishFixedIpTestCase(base.DefinitionBaseTestCase):
    extension_module = sn
    extension_attributes = (sn.DNS_PUBLISH_FIXED_IP,)