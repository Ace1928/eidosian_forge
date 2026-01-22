from oslo_db.sqlalchemy import test_fixtures
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_mitaka01(self, engine, data):
    indexes = get_indexes('images', engine)
    self.assertIn('created_at_image_idx', indexes)
    self.assertIn('updated_at_image_idx', indexes)