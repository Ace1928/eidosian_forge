from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_given_query_no_rows(self):
    specimen = MyModel(y='y1', z='z2', uuid='136254d5-3869-408f-9da7-190e0072641a')
    query = self.session.query(MyModel).filter_by(x=10)
    exc = self.assertRaises(update_match.NoRowsMatched, query.update_on_match, specimen, 'uuid', values={'x': 9, 'z': 'z3'})
    self.assertEqual('Zero rows matched for 3 attempts', exc.args[0])