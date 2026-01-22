from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class Test2024_1Expand01MySQL(Test2024_1Expand01Mixin, test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.MySQLOpportunisticFixture