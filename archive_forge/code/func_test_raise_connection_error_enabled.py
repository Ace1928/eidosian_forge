from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_raise_connection_error_enabled(self):
    self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True)
    self.test_db_api.error_counter = 5
    self.assertRaises(exception.DBConnectionError, self.dbapi.api_raise_default)
    self.assertEqual(4, self.test_db_api.error_counter, 'Unexpected retry')