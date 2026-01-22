import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefResourceTypeTests(object):

    def test_resource_type_get_all(self):
        resource_types_orig = self.db_api.metadef_resource_type_get_all(self.context)
        fixture = build_resource_type_fixture()
        self.db_api.metadef_resource_type_create(self.context, fixture)
        resource_types = self.db_api.metadef_resource_type_get_all(self.context)
        test_len = len(resource_types_orig) + 1
        self.assertEqual(test_len, len(resource_types))