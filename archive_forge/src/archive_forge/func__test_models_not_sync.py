from unittest import mock
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db.sqlalchemy import test_migrations as migrate
from oslo_db.tests.sqlalchemy import base as db_test_base
def _test_models_not_sync(self):
    self.metadata_migrations.clear()
    sa.Table('table', self.metadata_migrations, sa.Column('fk_check', sa.String(36), nullable=False), sa.PrimaryKeyConstraint('fk_check'), mysql_engine='InnoDB')
    sa.Table('testtbl', self.metadata_migrations, sa.Column('id', sa.Integer, primary_key=True), sa.Column('spam', sa.String(8), nullable=True), sa.Column('eggs', sa.DateTime), sa.Column('foo', sa.Boolean, server_default=sa.sql.expression.false()), sa.Column('bool_wo_default', sa.Boolean, unique=True), sa.Column('bar', sa.BigInteger), sa.Column('defaulttest', sa.Integer, server_default='7'), sa.Column('defaulttest2', sa.String(8), server_default=''), sa.Column('defaulttest3', sa.String(5), server_default='fake'), sa.Column('defaulttest4', sa.Enum('first', 'second', name='testenum'), server_default='first'), sa.Column('defaulttest5', sa.Integer, server_default=sa.text('0')), sa.Column('variant', sa.String(10)), sa.Column('fk_check', sa.String(36), nullable=False), sa.UniqueConstraint('spam', 'foo', name='uniq_cons'), sa.ForeignKeyConstraint(['fk_check'], ['table.fk_check']), mysql_engine='InnoDB')
    msg = str(self.assertRaises(AssertionError, self.test_models_sync))
    self.assertTrue(msg.startswith("Models and migration scripts aren't in sync:"))
    self.assertIn('testtbl', msg)
    self.assertIn('spam', msg)
    self.assertIn('eggs', msg)
    self.assertIn('foo', msg)
    self.assertIn('bar', msg)
    self.assertIn('bool_wo_default', msg)
    self.assertIn('defaulttest', msg)
    self.assertIn('defaulttest3', msg)
    self.assertIn('remove_fk', msg)
    self.assertIn('variant', msg)