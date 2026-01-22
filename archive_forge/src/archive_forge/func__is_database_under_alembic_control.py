import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from oslo_db import exception as db_exception
from oslo_log import log as logging
from oslo_utils import fileutils
from keystone.common import sql
import keystone.conf
def _is_database_under_alembic_control(engine):
    with engine.connect() as conn:
        context = alembic_migration.MigrationContext.configure(conn)
        return bool(context.get_current_heads())