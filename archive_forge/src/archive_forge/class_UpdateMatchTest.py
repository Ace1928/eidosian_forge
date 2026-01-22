from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class UpdateMatchTest(db_test_base._DbTestCase):

    def setUp(self):
        super(UpdateMatchTest, self).setUp()
        Base.metadata.create_all(self.engine)
        self.addCleanup(Base.metadata.drop_all, self.engine)
        self.session = self.sessionmaker(autocommit=False)
        self.addCleanup(self.session.close)
        self.session.add_all([MyModel(id=1, uuid='23cb9224-9f8e-40fe-bd3c-e7577b7af37d', x=5, y='y1', z='z1'), MyModel(id=2, uuid='136254d5-3869-408f-9da7-190e0072641a', x=6, y='y1', z='z2'), MyModel(id=3, uuid='094eb162-d5df-494b-a458-a91a1b2d2c65', x=7, y='y1', z='z1'), MyModel(id=4, uuid='94659b3f-ea1f-4ffd-998d-93b28f7f5b70', x=8, y='y2', z='z2'), MyModel(id=5, uuid='bdf3893c-ee3c-40a0-bc79-960adb6cd1d4', x=8, y='y2', z=None)])
        self.session.commit()

    def _assert_row(self, pk, values):
        row = self.session.execute(sql.select(MyModel.__table__).where(MyModel.__table__.c.id == pk)).first()
        values['id'] = pk
        self.assertEqual(values, dict(row._mapping))

    def test_update_specimen_successful(self):
        uuid = '136254d5-3869-408f-9da7-190e0072641a'
        specimen = MyModel(y='y1', z='z2', uuid=uuid)
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(2, result.id)
        self.assertEqual('z3', result.z)
        self.assertIn(result, self.session)
        self._assert_row(2, {'uuid': '136254d5-3869-408f-9da7-190e0072641a', 'x': 9, 'y': 'y1', 'z': 'z3'})

    def test_update_specimen_include_only(self):
        uuid = '136254d5-3869-408f-9da7-190e0072641a'
        specimen = MyModel(y='y9', z='z5', x=6, uuid=uuid)
        self.session.query(MyModel).filter(MyModel.uuid == uuid).one()
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'}, include_only=('x',))
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(2, result.id)
        self.assertEqual('z3', result.z)
        self.assertIn(result, self.session)
        self.assertNotIn(result, self.session.dirty)
        self._assert_row(2, {'uuid': '136254d5-3869-408f-9da7-190e0072641a', 'x': 9, 'y': 'y1', 'z': 'z3'})

    def test_update_specimen_no_rows(self):
        specimen = MyModel(y='y1', z='z3', uuid='136254d5-3869-408f-9da7-190e0072641a')
        exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).update_on_match, specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertEqual('Zero rows matched for 3 attempts', exc.args[0])

    def test_update_specimen_process_query_no_rows(self):
        specimen = MyModel(y='y1', z='z2', uuid='136254d5-3869-408f-9da7-190e0072641a')

        def process_query(query):
            return query.filter_by(x=10)
        exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).update_on_match, specimen, 'uuid', values={'x': 9, 'z': 'z3'}, process_query=process_query)
        self.assertEqual('Zero rows matched for 3 attempts', exc.args[0])

    def test_update_specimen_given_query_no_rows(self):
        specimen = MyModel(y='y1', z='z2', uuid='136254d5-3869-408f-9da7-190e0072641a')
        query = self.session.query(MyModel).filter_by(x=10)
        exc = self.assertRaises(update_match.NoRowsMatched, query.update_on_match, specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertEqual('Zero rows matched for 3 attempts', exc.args[0])

    def test_update_specimen_multi_rows(self):
        specimen = MyModel(y='y1', z='z1')
        exc = self.assertRaises(update_match.MultiRowsMatched, self.session.query(MyModel).update_on_match, specimen, 'y', values={'x': 9, 'z': 'z3'})
        self.assertEqual('2 rows matched; expected one', exc.args[0])

    def test_update_specimen_query_mismatch_error(self):
        specimen = MyModel(y='y1')
        q = self.session.query(MyModel.x, MyModel.y)
        exc = self.assertRaises(AssertionError, q.update_on_match, specimen, 'y', values={'x': 9, 'z': 'z3'})
        self.assertEqual('Query does not match given specimen', exc.args[0])

    def test_custom_handle_failure_raise_new(self):

        class MyException(Exception):
            pass

        def handle_failure(query):
            result = query.count()
            self.assertEqual(0, result)
            raise MyException('test: %d' % result)
        specimen = MyModel(y='y1', z='z3', uuid='136254d5-3869-408f-9da7-190e0072641a')
        exc = self.assertRaises(MyException, self.session.query(MyModel).update_on_match, specimen, 'uuid', values={'x': 9, 'z': 'z3'}, handle_failure=handle_failure)
        self.assertEqual('test: 0', exc.args[0])

    def test_custom_handle_failure_cancel_raise(self):
        uuid = '136254d5-3869-408f-9da7-190e0072641a'

        class MyException(Exception):
            pass

        def handle_failure(query):
            result = query.count()
            self.assertEqual(0, result)
            return True
        specimen = MyModel(id=2, y='y1', z='z3', uuid=uuid)
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'}, handle_failure=handle_failure)
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(2, result.id)
        self.assertEqual('z3', result.z)
        self.assertEqual(9, result.x)
        self.assertIn(result, self.session)

    def test_update_specimen_on_none_successful(self):
        uuid = 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4'
        specimen = MyModel(y='y2', z=None, uuid=uuid)
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertIn(result, self.session)
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(5, result.id)
        self.assertEqual('z3', result.z)
        self._assert_row(5, {'uuid': 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4', 'x': 9, 'y': 'y2', 'z': 'z3'})

    def test_update_specimen_on_multiple_nonnone_successful(self):
        uuid = '094eb162-d5df-494b-a458-a91a1b2d2c65'
        specimen = MyModel(y=('y1', 'y2'), x=(5, 7), uuid=uuid)
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertIn(result, self.session)
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(3, result.id)
        self.assertEqual('z3', result.z)
        self._assert_row(3, {'uuid': '094eb162-d5df-494b-a458-a91a1b2d2c65', 'x': 9, 'y': 'y1', 'z': 'z3'})

    def test_update_specimen_on_multiple_wnone_successful(self):
        uuid = 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4'
        specimen = MyModel(y=('y1', 'y2'), x=(8, 7), z=('z1', 'z2', None), uuid=uuid)
        result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'})
        self.assertIn(result, self.session)
        self.assertEqual(uuid, result.uuid)
        self.assertEqual(5, result.id)
        self.assertEqual('z3', result.z)
        self._assert_row(5, {'uuid': 'bdf3893c-ee3c-40a0-bc79-960adb6cd1d4', 'x': 9, 'y': 'y2', 'z': 'z3'})

    def test_update_returning_pk_matched(self):
        pk = self.session.query(MyModel).filter_by(y='y1', z='z2').update_returning_pk({'x': 9, 'z': 'z3'}, ('uuid', '136254d5-3869-408f-9da7-190e0072641a'))
        self.assertEqual((2,), pk)
        self._assert_row(2, {'uuid': '136254d5-3869-408f-9da7-190e0072641a', 'x': 9, 'y': 'y1', 'z': 'z3'})

    def test_update_returning_wrong_uuid(self):
        exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z2').update_returning_pk, {'x': 9, 'z': 'z3'}, ('uuid', '23cb9224-9f8e-40fe-bd3c-e7577b7af37d'))
        self.assertEqual('No rows matched the UPDATE', exc.args[0])

    def test_update_returning_no_rows(self):
        exc = self.assertRaises(update_match.NoRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z3').update_returning_pk, {'x': 9, 'z': 'z3'}, ('uuid', '136254d5-3869-408f-9da7-190e0072641a'))
        self.assertEqual('No rows matched the UPDATE', exc.args[0])

    def test_update_multiple_rows(self):
        exc = self.assertRaises(update_match.MultiRowsMatched, self.session.query(MyModel).filter_by(y='y1', z='z1').update_returning_pk, {'x': 9, 'z': 'z3'}, ('y', 'y1'))
        self.assertEqual('2 rows matched; expected one', exc.args[0])