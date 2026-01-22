import sqlalchemy
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
class BaseBackendSqlTests(unit.SQLDriverOverrides, unit.TestCase):

    def setUp(self):
        super().setUp()
        self.database_fixture = self.useFixture(database.Database())
        self.load_backends()
        self.load_fixtures(default_fixtures)
        self.user_foo['enabled'] = True

    def config_files(self):
        config_files = super(BaseBackendSqlTests, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files