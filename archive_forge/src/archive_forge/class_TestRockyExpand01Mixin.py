from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestRockyExpand01Mixin(test_migrations.AlembicMigrationsMixin):

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='rocky_expand01')

    def _pre_upgrade_rocky_expand01(self, engine):
        images = db_utils.get_table(engine, 'images')
        self.assertNotIn('os_hidden', images.c)

    def _check_rocky_expand01(self, engine, data):
        images = db_utils.get_table(engine, 'images')
        self.assertIn('os_hidden', images.c)
        self.assertFalse(images.c.os_hidden.nullable)