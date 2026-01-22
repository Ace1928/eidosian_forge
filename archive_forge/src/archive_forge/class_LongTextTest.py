from sqlalchemy.dialects.mysql import base as mysql_base
from sqlalchemy.dialects.sqlite import base as sqlite_base
from sqlalchemy import types
from heat.db import types as db_types
from heat.tests import common
class LongTextTest(common.HeatTestCase):

    def setUp(self):
        super(LongTextTest, self).setUp()
        self.sqltype = db_types.LongText()

    def test_load_dialect_impl(self):
        dialect = mysql_base.MySQLDialect()
        impl = self.sqltype.load_dialect_impl(dialect)
        self.assertNotEqual(types.Text, type(impl))
        dialect = sqlite_base.SQLiteDialect()
        impl = self.sqltype.load_dialect_impl(dialect)
        self.assertEqual(types.Text, type(impl))