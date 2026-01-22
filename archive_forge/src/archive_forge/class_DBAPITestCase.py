from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
class DBAPITestCase(test_base.BaseTestCase):

    def test_dbapi_full_path_module_method(self):
        dbapi = api.DBAPI('oslo_db.tests.test_api')
        result = dbapi.api_class_call1(1, 2, kwarg1='meow')
        expected = ((1, 2), {'kwarg1': 'meow'})
        self.assertEqual(expected, result)

    def test_dbapi_unknown_invalid_backend(self):
        self.assertRaises(ImportError, api.DBAPI, 'tests.unit.db.not_existent')

    def test_dbapi_lazy_loading(self):
        dbapi = api.DBAPI('oslo_db.tests.test_api', lazy=True)
        self.assertIsNone(dbapi._backend)
        dbapi.api_class_call1(1, 'abc')
        self.assertIsNotNone(dbapi._backend)

    def test_dbapi_from_config(self):
        conf = cfg.ConfigOpts()
        dbapi = api.DBAPI.from_config(conf, backend_mapping={'sqlalchemy': __name__})
        self.assertIsNotNone(dbapi._backend)