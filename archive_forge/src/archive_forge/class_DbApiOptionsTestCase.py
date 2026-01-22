from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_db import options
from oslo_db.tests import base as test_base
class DbApiOptionsTestCase(test_base.BaseTestCase):

    def setUp(self):
        super().setUp()
        self.conf = self.useFixture(config_fixture.Config()).conf
        self.conf.register_opts(options.database_opts, group='database')

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

    def test_dbapi_parameters(self):
        path = self.create_tempfiles([['tmp', b'[database]\nbackend=test_123\n']])[0]
        self.conf(['--config-file', path])
        self.assertEqual('test_123', self.conf.database.backend)

    def test_set_defaults(self):
        conf = cfg.ConfigOpts()
        options.set_defaults(conf, connection='sqlite:///:memory:')
        self.assertTrue(len(conf.database.items()) > 1)
        self.assertEqual('sqlite:///:memory:', conf.database.connection)
        self.assertEqual(None, self.conf.database.mysql_wsrep_sync_wait)