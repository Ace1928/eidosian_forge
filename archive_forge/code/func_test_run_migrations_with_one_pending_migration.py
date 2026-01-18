from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
def test_run_migrations_with_one_pending_migration(self):
    zebra1 = mock.Mock()
    zebra1.has_migrations.return_value = False
    zebra1.migrate.return_value = 0
    zebra2 = mock.Mock()
    zebra2.has_migrations.return_value = True
    zebra2.migrate.return_value = 50
    migrations = [zebra1, zebra2]
    engine = mock.Mock()
    actual = data_migrations._run_migrations(engine, migrations)
    self.assertEqual(50, actual)
    zebra1.has_migrations.assert_called_once_with(engine)
    zebra1.migrate.assert_not_called()
    zebra2.has_migrations.assert_called_once_with(engine)
    zebra2.migrate.assert_called_once_with(engine)