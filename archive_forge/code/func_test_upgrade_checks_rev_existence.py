from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_upgrade_checks_rev_existence(self):
    self.first_ext.obj.has_revision.return_value = False
    self.second_ext.obj.has_revision.return_value = False
    self.assertRaises(exception.DBMigrationError, self.migration_manager.upgrade, 100)
    self.assertEqual([100, 200], self.migration_manager.upgrade(None))
    self.second_ext.obj.has_revision.return_value = True
    self.assertEqual([100, 200], self.migration_manager.upgrade(200))
    self.assertEqual([100, 200], self.migration_manager.upgrade(None))