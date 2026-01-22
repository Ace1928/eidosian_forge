import os
import alembic
from alembic import config as alembic_config
import alembic.migration as alembic_migration
from alembic import script as alembic_script
from oslo_db.sqlalchemy.migration_cli import ext_base
Stamps database with provided revision.

        :param revision: Should match one from repository or head - to stamp
                         database with most recent revision
        :type revision: string
        