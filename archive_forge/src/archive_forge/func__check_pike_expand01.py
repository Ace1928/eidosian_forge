from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_pike_expand01(self, engine, data):
    self._pre_upgrade_pike_expand01(engine)