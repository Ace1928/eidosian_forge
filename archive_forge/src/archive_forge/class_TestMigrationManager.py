from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
class TestMigrationManager(test_base.BaseTestCase):

    def setUp(self):
        self.migration_config = {'alembic_ini_path': '.', 'migrate_repo_path': '.', 'db_url': 'sqlite://'}
        engine = sqlalchemy.create_engine(self.migration_config['db_url'])
        self.migration_manager = manager.MigrationManager(self.migration_config, engine)
        self.ext = mock.Mock()
        self.ext.obj.version = mock.Mock(return_value=0)
        self.migration_manager._manager.extensions = [self.ext]
        super(TestMigrationManager, self).setUp()

    def test_manager_update(self):
        self.migration_manager.upgrade('head')
        self.ext.obj.upgrade.assert_called_once_with('head')

    def test_manager_update_revision_none(self):
        self.migration_manager.upgrade(None)
        self.ext.obj.upgrade.assert_called_once_with(None)

    def test_downgrade_normal_revision(self):
        self.migration_manager.downgrade('111abcd')
        self.ext.obj.downgrade.assert_called_once_with('111abcd')

    def test_version(self):
        self.migration_manager.version()
        self.ext.obj.version.assert_called_once_with()

    def test_version_return_value(self):
        version = self.migration_manager.version()
        self.assertEqual(0, version)

    def test_revision_message_autogenerate(self):
        self.migration_manager.revision('test', True)
        self.ext.obj.revision.assert_called_once_with('test', True)

    def test_revision_only_message(self):
        self.migration_manager.revision('test', False)
        self.ext.obj.revision.assert_called_once_with('test', False)

    def test_stamp(self):
        self.migration_manager.stamp('stamp')
        self.ext.obj.stamp.assert_called_once_with('stamp')

    def test_wrong_config(self):
        err = self.assertRaises(ValueError, manager.MigrationManager, {'wrong_key': 'sqlite://'})
        self.assertEqual('Either database url or engine must be provided.', err.args[0])