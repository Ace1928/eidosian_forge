from sqlalchemy.dialects.mysql import base as mysql_base
from sqlalchemy.dialects.sqlite import base as sqlite_base
from sqlalchemy import types
from heat.db import types as db_types
from heat.tests import common
def test_process_bind_param(self):
    dialect = None
    value = ['foo', 'bar']
    result = self.sqltype.process_bind_param(value, dialect)
    self.assertEqual('["foo", "bar"]', result)