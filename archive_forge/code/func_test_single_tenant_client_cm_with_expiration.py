from unittest import mock
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import store as swift_store
from glance_store import exceptions
from glance_store.tests import base
def test_single_tenant_client_cm_with_expiration(self):
    store = self.prepare_store()
    manager = connection_manager.SingleTenantConnectionManager(store=store, store_location=self.location, allow_reauth=True)
    store.init_client.assert_called_once_with(self.location, None)
    self.client.session.get_endpoint.assert_called_once_with(service_type=store.service_type, interface=store.endpoint_type, region_name=store.region)
    auth_ref = mock.MagicMock()
    self.client.session.auth.get_auth_ref.return_value = auth_ref
    auth_ref.will_expire_soon.return_value = True
    manager.get_connection()
    self.assertEqual(2, store.get_store_connection.call_count)
    self.assertEqual(2, self.client.session.get_auth_headers.call_count)