from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_update_specimen_on_multiple_nonnone_successful(self):
    uuid = '094eb162-d5df-494b-a458-a91a1b2d2c65'
    specimen = MyModel(y=('y1', 'y2'), x=(5, 7), uuid=uuid)
    result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
    self.assertIn(result, self.session)
    self.assertEqual(uuid, result.uuid)
    self.assertEqual(3, result.id)
    self.assertEqual('z3', result.z)
    self._assert_row(3, {'uuid': '094eb162-d5df-494b-a458-a91a1b2d2c65', 'x': 9, 'y': 'y1', 'z': 'z3'})