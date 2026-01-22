from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy.test_base import backend_specific  # noqa
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslo_db.tests import base as test_base
class _DbTestCase(db_fixtures.OpportunisticDBTestMixin, test_base.BaseTestCase):

    def setUp(self):
        super(_DbTestCase, self).setUp()
        self.engine = enginefacade.writer.get_engine()
        self.sessionmaker = enginefacade.writer.get_sessionmaker()