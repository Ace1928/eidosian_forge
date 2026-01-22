from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import _proxy
from openstack.container_infrastructure_management.v1 import cluster
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack.container_infrastructure_management.v1 import service
from openstack.tests.unit import test_proxy_base
class TestClusterCertificate(TestMagnumProxy):

    def test_cluster_certificate_get(self):
        self.verify_get(self.proxy.get_cluster_certificate, cluster_certificate.ClusterCertificate)

    def test_cluster_certificate_create_attrs(self):
        self.verify_create(self.proxy.create_cluster_certificate, cluster_certificate.ClusterCertificate)