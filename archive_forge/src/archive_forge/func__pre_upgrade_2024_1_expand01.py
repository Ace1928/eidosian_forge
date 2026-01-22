from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _pre_upgrade_2024_1_expand01(self, engine):
    self.assertRaises(sqlalchemy.exc.NoSuchTableError, db_utils.get_table, engine, 'node_reference')
    self.assertRaises(sqlalchemy.exc.NoSuchTableError, db_utils.get_table, engine, 'cached_images')