import collections
import os
from alembic import command as alembic_command
from alembic import script as alembic_script
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from sqlalchemy import sql
import sqlalchemy.types as types
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import versions
from glance.db.sqlalchemy import models
from glance.db.sqlalchemy import models_metadef
import glance.tests.utils as test_utils
class AlembicMigrationsMixin(object):

    def setUp(self):
        super(AlembicMigrationsMixin, self).setUp()
        self.engine = enginefacade.writer.get_engine()

    def _get_revisions(self, config, head=None):
        head = head or 'heads'
        scripts_dir = alembic_script.ScriptDirectory.from_config(config)
        revisions = list(scripts_dir.walk_revisions(base='base', head=head))
        revisions = list(reversed(revisions))
        revisions = [rev.revision for rev in revisions]
        return revisions

    def _migrate_up(self, config, engine, revision, with_data=False):
        if with_data:
            data = None
            pre_upgrade = getattr(self, '_pre_upgrade_%s' % revision, None)
            if pre_upgrade:
                data = pre_upgrade(engine)
        alembic_command.upgrade(config, revision)
        if with_data:
            check = getattr(self, '_check_%s' % revision, None)
            if check:
                check(engine, data)

    def test_walk_versions(self):
        alembic_config = alembic_migrations.get_alembic_config(self.engine)
        for revision in self._get_revisions(alembic_config):
            self._migrate_up(alembic_config, self.engine, revision, with_data=True)