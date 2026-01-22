import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class TestMetadefDriver(test_utils.BaseTestCase):
    """Test Driver class for Metadef tests."""

    def setUp(self):
        """Run before each test method to initialize test environment."""
        super(TestMetadefDriver, self).setUp()
        config.parse_args(args=[])
        self.config(metadata_source_path=METADEFS_DIR)
        context_cls = context.RequestContext
        self.adm_context = context_cls(is_admin=True, auth_token='user:user:admin')
        self.context = context_cls(is_admin=False, auth_token='user:user:user')
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)

    def _assert_saved_fields(self, expected, actual):
        for k in expected.keys():
            self.assertEqual(expected[k], actual[k])