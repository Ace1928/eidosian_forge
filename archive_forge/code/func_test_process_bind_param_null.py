from sqlalchemy.dialects.mysql import base as mysql_base
from sqlalchemy.dialects.sqlite import base as sqlite_base
from sqlalchemy import types
from heat.db import types as db_types
from heat.tests import common
def test_process_bind_param_null(self):
    dialect = None
    value = None
    result = self.sqltype.process_bind_param(value, dialect)
    self.assertEqual('null', result)