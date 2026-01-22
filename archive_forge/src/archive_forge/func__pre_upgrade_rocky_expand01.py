from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _pre_upgrade_rocky_expand01(self, engine):
    images = db_utils.get_table(engine, 'images')
    self.assertNotIn('os_hidden', images.c)