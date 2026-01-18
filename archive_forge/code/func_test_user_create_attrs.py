from openstack.identity.v2 import _proxy
from openstack.identity.v2 import role
from openstack.identity.v2 import tenant
from openstack.identity.v2 import user
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_user_create_attrs(self):
    self.verify_create(self.proxy.create_user, user.User)