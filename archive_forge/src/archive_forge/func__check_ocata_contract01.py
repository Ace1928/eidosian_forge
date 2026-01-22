import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_ocata_contract01(self, engine, data):
    images = db_utils.get_table(engine, 'images')
    self.assertNotIn('is_public', images.c)
    self.assertIn('visibility', images.c)