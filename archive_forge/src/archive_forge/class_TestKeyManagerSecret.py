from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
class TestKeyManagerSecret(TestKeyManagerProxy):

    def test_secret_create_attrs(self):
        self.verify_create(self.proxy.create_secret, secret.Secret)

    def test_secret_delete(self):
        self.verify_delete(self.proxy.delete_secret, secret.Secret, False)

    def test_secret_delete_ignore(self):
        self.verify_delete(self.proxy.delete_secret, secret.Secret, True)

    def test_secret_find(self):
        self.verify_find(self.proxy.find_secret, secret.Secret)

    def test_secret_get(self):
        self.verify_get(self.proxy.get_secret, secret.Secret)
        self.verify_get_overrided(self.proxy, secret.Secret, 'openstack.key_manager.v1.secret.Secret')

    def test_secrets(self):
        self.verify_list(self.proxy.secrets, secret.Secret)

    def test_secret_update(self):
        self.verify_update(self.proxy.update_secret, secret.Secret)