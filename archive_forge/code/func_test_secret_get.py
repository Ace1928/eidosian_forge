from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
def test_secret_get(self):
    self.verify_get(self.proxy.get_secret, secret.Secret)
    self.verify_get_overrided(self.proxy, secret.Secret, 'openstack.key_manager.v1.secret.Secret')