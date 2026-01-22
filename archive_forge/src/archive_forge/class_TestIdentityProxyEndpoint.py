import uuid
from openstack.identity.v3 import _proxy
from openstack.identity.v3 import access_rule
from openstack.identity.v3 import credential
from openstack.identity.v3 import domain
from openstack.identity.v3 import domain_config
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import group
from openstack.identity.v3 import policy
from openstack.identity.v3 import project
from openstack.identity.v3 import region
from openstack.identity.v3 import role
from openstack.identity.v3 import role_domain_group_assignment
from openstack.identity.v3 import role_domain_user_assignment
from openstack.identity.v3 import role_project_group_assignment
from openstack.identity.v3 import role_project_user_assignment
from openstack.identity.v3 import role_system_group_assignment
from openstack.identity.v3 import role_system_user_assignment
from openstack.identity.v3 import service
from openstack.identity.v3 import trust
from openstack.identity.v3 import user
from openstack.tests.unit import test_proxy_base
class TestIdentityProxyEndpoint(TestIdentityProxyBase):

    def test_endpoint_create_attrs(self):
        self.verify_create(self.proxy.create_endpoint, endpoint.Endpoint)

    def test_endpoint_delete(self):
        self.verify_delete(self.proxy.delete_endpoint, endpoint.Endpoint, False)

    def test_endpoint_delete_ignore(self):
        self.verify_delete(self.proxy.delete_endpoint, endpoint.Endpoint, True)

    def test_endpoint_find(self):
        self.verify_find(self.proxy.find_endpoint, endpoint.Endpoint)

    def test_endpoint_get(self):
        self.verify_get(self.proxy.get_endpoint, endpoint.Endpoint)

    def test_endpoints(self):
        self.verify_list(self.proxy.endpoints, endpoint.Endpoint)

    def test_endpoint_update(self):
        self.verify_update(self.proxy.update_endpoint, endpoint.Endpoint)