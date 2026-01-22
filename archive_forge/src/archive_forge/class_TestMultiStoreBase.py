from unittest import mock
from oslo_config import cfg
import glance_store as store
from glance_store import backend
from glance_store import location
from glance_store import multi_backend
from glance_store.tests import base
class TestMultiStoreBase(base.MultiStoreBaseTest):
    _CONF = multi_backend.CONF

    def setUp(self):
        super(TestMultiStoreBase, self).setUp()
        enabled_backends = {'fast': 'file', 'cheap': 'file'}
        self.reserved_stores = {'consuming_service_reserved_store': 'file'}
        self.conf = self._CONF
        self.conf(args=[])
        self.conf.register_opt(cfg.DictOpt('enabled_backends'))
        self.config(enabled_backends=enabled_backends)
        store.register_store_opts(self.conf, reserved_stores=self.reserved_stores)
        self.config(default_backend='file1', group='glance_store')
        location.SCHEME_TO_CLS_BACKEND_MAP = {}
        store.create_multi_stores(self.conf, reserved_stores=self.reserved_stores)
        self.addCleanup(setattr, location, 'SCHEME_TO_CLS_BACKEND_MAP', dict())
        self.addCleanup(self.conf.reset)

    def test_reserved_stores_loaded(self):
        store = multi_backend.get_store_from_store_identifier('consuming_service_reserved_store')
        self.assertIsNotNone(store)
        self.assertEqual(self.reserved_stores, multi_backend._RESERVED_STORES)
        self.assertEqual('consuming_service_reserved_store', store.backend_group)