import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
This mixin is used to create an initial glance database and upgrade it
    up to the train_expand01 revision.
    