from openstack.identity.v2 import _proxy
from openstack.identity.v2 import role
from openstack.identity.v2 import tenant
from openstack.identity.v2 import user
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_user_delete_ignore(self):
    self.verify_delete(self.proxy.delete_user, user.User, True)