import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestTrainMigrate01_EmptyDBMySQL(TestTrainMigrate01_EmptyDBMixin, test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.MySQLOpportunisticFixture