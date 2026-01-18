from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_multi_rows(self):
    specimen = MyModel(y='y1', z='z1')
    exc = self.assertRaises(update_match.MultiRowsMatched, self.session.query(MyModel).update_on_match, specimen, 'y', values={'x': 9, 'z': 'z3'})
    self.assertEqual('2 rows matched; expected one', exc.args[0])