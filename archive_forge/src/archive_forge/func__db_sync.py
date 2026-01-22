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
def _db_sync(branch=None, *, engine=None):
    config = _find_alembic_conf()
    if engine is None:
        with sql.session_for_write() as session:
            engine = session.get_bind()
    engine_url = str(engine.url).replace('%', '%%')
    config.set_main_option('sqlalchemy.url', str(engine_url))
    _upgrade_alembic(engine, config, branch)