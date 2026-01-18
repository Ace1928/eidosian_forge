from openstack.container_infrastructure_management.v1 import (
from openstack.tests.unit import base
def test_sign_coe_cluster_certificate(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='certificates'), json={'cluster_uuid': coe_cluster_signed_cert_obj['cluster_uuid'], 'csr': coe_cluster_signed_cert_obj['csr']})])
    self.cloud.sign_coe_cluster_certificate(coe_cluster_signed_cert_obj['cluster_uuid'], coe_cluster_signed_cert_obj['csr'])
    self.assert_calls()