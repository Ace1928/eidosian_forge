from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_wallaby_expand01(self, engine, data):
    tasks = db_utils.get_table(engine, 'tasks')
    self.assertIn('image_id', tasks.c)
    self.assertIn('request_id', tasks.c)
    self.assertIn('user_id', tasks.c)
    self.assertTrue(tasks.c.image_id.nullable)
    self.assertTrue(tasks.c.request_id.nullable)
    self.assertTrue(tasks.c.user_id.nullable)
    self.assertTrue(db_utils.index_exists(engine, 'tasks', 'ix_tasks_image_id'), 'Index %s on table %s does not exist' % ('ix_tasks_image_id', 'tasks'))