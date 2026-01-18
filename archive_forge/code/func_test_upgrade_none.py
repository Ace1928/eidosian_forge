from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_upgrade_none(self, command):
    self.alembic.upgrade(None)
    command.upgrade.assert_called_once_with(self.alembic.config, 'head')