from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestRockyExpand01MySQL(TestRockyExpand01Mixin, test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.MySQLOpportunisticFixture