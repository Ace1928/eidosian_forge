from sqlalchemy import Column, Integer
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import types
from oslo_db.tests.sqlalchemy import base as test_base
class JsonTypesTestCase(test_base._DbTestCase):

    def setUp(self):
        super(JsonTypesTestCase, self).setUp()
        JsonTable.__table__.create(self.engine)
        self.addCleanup(JsonTable.__table__.drop, self.engine)
        self.session = self.sessionmaker()
        self.addCleanup(self.session.close)

    def test_default_value(self):
        with self.session.begin():
            JsonTable(id=1).save(self.session)
        obj = self.session.query(JsonTable).filter_by(id=1).one()
        self.assertEqual([], obj.jlist)
        self.assertEqual({}, obj.jdict)
        self.assertIsNone(obj.json)

    def test_dict(self):
        test = {'a': 42, 'b': [1, 2, 3]}
        with self.session.begin():
            JsonTable(id=1, jdict=test).save(self.session)
        obj = self.session.query(JsonTable).filter_by(id=1).one()
        self.assertEqual(test, obj.jdict)

    def test_list(self):
        test = [1, True, 'hello', {}]
        with self.session.begin():
            JsonTable(id=1, jlist=test).save(self.session)
        obj = self.session.query(JsonTable).filter_by(id=1).one()
        self.assertEqual(test, obj.jlist)

    def test_dict_type_check(self):
        self.assertRaises(db_exc.DBError, JsonTable(id=1, jdict=[]).save, self.session)

    def test_list_type_check(self):
        self.assertRaises(db_exc.DBError, JsonTable(id=1, jlist={}).save, self.session)

    def test_generic(self):
        tested = ['string', 42, True, None, [1, 2, 3], {'a': 'b'}]
        for i, test in enumerate(tested):
            JsonTable(id=i, json=test).save(self.session)
            obj = self.session.query(JsonTable).filter_by(id=i).one()
            self.assertEqual(test, obj.json)

    def test_mysql_variants(self):
        self.assertEqual('LONGTEXT', str(types.JsonEncodedDict(mysql_as_long=True).compile(dialect=mysql.dialect())))
        self.assertEqual('MEDIUMTEXT', str(types.JsonEncodedDict(mysql_as_medium=True).compile(dialect=mysql.dialect())))
        self.assertRaises(TypeError, lambda: types.JsonEncodedDict(mysql_as_long=True, mysql_as_medium=True))