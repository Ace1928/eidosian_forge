from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_db import options
from oslo_db.tests import base as test_base
def test_session_parameters(self):
    path = self.create_tempfiles([['tmp', b'[database]\nconnection=x://y.z\nmax_pool_size=20\nmax_retries=30\nretry_interval=40\nmax_overflow=50\nconnection_debug=60\nconnection_trace=True\npool_timeout=7\n']])[0]
    self.conf(['--config-file', path])
    self.assertEqual('x://y.z', self.conf.database.connection)
    self.assertEqual(20, self.conf.database.max_pool_size)
    self.assertEqual(30, self.conf.database.max_retries)
    self.assertEqual(40, self.conf.database.retry_interval)
    self.assertEqual(50, self.conf.database.max_overflow)
    self.assertEqual(60, self.conf.database.connection_debug)
    self.assertEqual(True, self.conf.database.connection_trace)
    self.assertEqual(7, self.conf.database.pool_timeout)