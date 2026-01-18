import io
from unittest import mock
import fixtures
from glance.cmd import manage
from glance.common import exception
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata as db_metadata
from glance.tests import utils as test_utils
from sqlalchemy.engine.url import make_url as sqlalchemy_make_url
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_current_alembic_heads')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_alembic_branch_head')
@mock.patch.object(manage.DbCommands, '_validate_engine')
@mock.patch.object(manage.DbCommands, 'expand')
def test_sync_failed_to_sync(self, mock_expand, mock_validate_engine, mock_get_alembic_branch_head, mock_get_current_alembic_heads):
    engine = mock_validate_engine.return_value
    engine.engine.name = 'mysql'
    mock_get_current_alembic_heads.return_value = ['ocata_contract01']
    mock_get_alembic_branch_head.side_effect = ['pike_contract01', '']
    mock_expand.side_effect = exception.GlanceException
    exit = self.assertRaises(SystemExit, self.db.sync)
    self.assertIn('Failed to sync database: ERROR:', exit.code)