import sqlalchemy
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
class BaseBackendSqlModels(BaseBackendSqlTests):

    def load_table(self, name):
        table = sqlalchemy.Table(name, sql.ModelBase.metadata, autoload_with=self.database_fixture.engine)
        return table

    def assertExpectedSchema(self, table, cols):
        table = self.load_table(table)
        for col, type_, length in cols:
            self.assertIsInstance(table.c[col].type, type_)
            if length:
                self.assertEqual(length, table.c[col].type.length)