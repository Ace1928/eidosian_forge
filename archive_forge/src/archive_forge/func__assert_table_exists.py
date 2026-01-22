import os
import sys
from oslo_config import cfg
from oslo_db import options as db_options
from glance.common import utils
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests import functional
from glance.tests.utils import depends_on_exe
from glance.tests.utils import execute
from glance.tests.utils import skip_if_disabled
def _assert_table_exists(self, db_table):
    cmd = 'sqlite3 {0} "SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'{1}\'"'.format(self.db_filepath, db_table)
    exitcode, out, err = execute(cmd, raise_error=True)
    msg = 'Expected table {0} was not found in the schema'.format(db_table)
    self.assertEqual(out.rstrip().decode('utf-8'), db_table, msg)