from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_query_mismatch_error(self):
    specimen = MyModel(y='y1')
    q = self.session.query(MyModel.x, MyModel.y)
    exc = self.assertRaises(AssertionError, q.update_on_match, specimen, 'y', values={'x': 9, 'z': 'z3'})
    self.assertEqual('Query does not match given specimen', exc.args[0])