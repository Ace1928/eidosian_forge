from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class Test2024_1Expand01Mixin(test_migrations.AlembicMigrationsMixin):

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='2024_1_expand01')

    def _pre_upgrade_2024_1_expand01(self, engine):
        self.assertRaises(sqlalchemy.exc.NoSuchTableError, db_utils.get_table, engine, 'node_reference')
        self.assertRaises(sqlalchemy.exc.NoSuchTableError, db_utils.get_table, engine, 'cached_images')

    def _check_2024_1_expand01(self, engine, data):
        node_reference = db_utils.get_table(engine, 'node_reference')
        self.assertIn('node_reference_id', node_reference.c)
        self.assertIn('node_reference_url', node_reference.c)
        self.assertTrue(db_utils.index_exists(engine, 'node_reference', 'uq_node_reference_node_reference_url'), 'Index %s on table %s does not exist' % ('uq_node_reference_node_reference_url', 'node_reference'))
        cached_images = db_utils.get_table(engine, 'cached_images')
        self.assertIn('id', cached_images.c)
        self.assertIn('image_id', cached_images.c)
        self.assertIn('last_accessed', cached_images.c)
        self.assertIn('last_modified', cached_images.c)
        self.assertIn('size', cached_images.c)
        self.assertIn('hits', cached_images.c)
        self.assertIn('checksum', cached_images.c)
        self.assertIn('node_reference_id', cached_images.c)
        self.assertTrue(db_utils.index_exists(engine, 'cached_images', 'ix_cached_images_image_id_node_reference_id'), 'Index %s on table %s does not exist' % ('ix_cached_images_image_id_node_reference_id', 'cached_images'))