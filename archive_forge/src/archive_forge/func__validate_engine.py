import os
import sys
import time
from alembic import command as alembic_command
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_log import log as logging
from oslo_utils import encodeutils
from glance.common import config
from glance.common import exception
from glance import context
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata
from glance.i18n import _
def _validate_engine(self, engine):
    """Check engine is valid or not.

        MySql is only supported for online upgrade.
        Adding sqlite as engine to support existing functional test cases.

        :param engine: database engine name
        """
    if engine.engine.name not in ['mysql', 'sqlite']:
        sys.exit(_('Rolling upgrades are currently supported only for MySQL and Sqlite'))