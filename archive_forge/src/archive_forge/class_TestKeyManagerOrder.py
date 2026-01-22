from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
class TestKeyManagerOrder(TestKeyManagerProxy):

    def test_order_create_attrs(self):
        self.verify_create(self.proxy.create_order, order.Order)

    def test_order_delete(self):
        self.verify_delete(self.proxy.delete_order, order.Order, False)

    def test_order_delete_ignore(self):
        self.verify_delete(self.proxy.delete_order, order.Order, True)

    def test_order_find(self):
        self.verify_find(self.proxy.find_order, order.Order)

    def test_order_get(self):
        self.verify_get(self.proxy.get_order, order.Order)

    def test_orders(self):
        self.verify_list(self.proxy.orders, order.Order)

    def test_order_update(self):
        self.verify_update(self.proxy.update_order, order.Order)