import os
from unittest import mock
from sqlalchemy.engine import url as sqla_url
from sqlalchemy import exc as sa_exc
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import types
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class DropAllObjectsTest(db_test_base._DbTestCase):

    def setUp(self):
        super(DropAllObjectsTest, self).setUp()
        self.metadata = metadata = schema.MetaData()
        schema.Table('a', metadata, schema.Column('id', types.Integer, primary_key=True), mysql_engine='InnoDB')
        schema.Table('b', metadata, schema.Column('id', types.Integer, primary_key=True), schema.Column('a_id', types.Integer, schema.ForeignKey('a.id')), mysql_engine='InnoDB')
        schema.Table('c', metadata, schema.Column('id', types.Integer, primary_key=True), schema.Column('b_id', types.Integer, schema.ForeignKey('b.id')), schema.Column('d_id', types.Integer, schema.ForeignKey('d.id', use_alter=True, name='c_d_fk')), mysql_engine='InnoDB')
        schema.Table('d', metadata, schema.Column('id', types.Integer, primary_key=True), schema.Column('c_id', types.Integer, schema.ForeignKey('c.id')), mysql_engine='InnoDB')
        metadata.create_all(self.engine, checkfirst=False)
        self.addCleanup(metadata.drop_all, self.engine, checkfirst=True)

    def test_drop_all(self):
        insp = inspect(self.engine)
        self.assertEqual(set(['a', 'b', 'c', 'd']), set(insp.get_table_names()))
        self._get_default_provisioned_db().backend.drop_all_objects(self.engine)
        insp = inspect(self.engine)
        self.assertEqual([], insp.get_table_names())