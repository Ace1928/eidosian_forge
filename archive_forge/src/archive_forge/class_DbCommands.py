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
class DbCommands(object):
    """Class for managing the db"""

    def __init__(self):
        pass

    def version(self):
        """Print database's current migration level"""
        current_heads = alembic_migrations.get_current_alembic_heads()
        if current_heads:
            for head in current_heads:
                print(head)
        else:
            print(_('Database is either not under migration control or under legacy migration control, please run "glance-manage db sync" to place the database under alembic migration control.'))

    def check(self):
        """Report any pending database upgrades.

        An exit code of 3 indicates db expand is needed, see stdout output.
        An exit code of 4 indicates db migrate is needed, see stdout output.
        An exit code of 5 indicates db contract is needed, see stdout output.
        """
        engine = db_api.get_engine()
        self._validate_engine(engine)
        curr_heads = alembic_migrations.get_current_alembic_heads()
        expand_heads = alembic_migrations.get_alembic_branch_head(db_migration.EXPAND_BRANCH)
        contract_heads = alembic_migrations.get_alembic_branch_head(db_migration.CONTRACT_BRANCH)
        if contract_heads in curr_heads:
            print(_('Database is up to date. No upgrades needed.'))
            sys.exit()
        elif not expand_heads or expand_heads not in curr_heads:
            print(_('Your database is not up to date. Your first step is to run `glance-manage db expand`.'))
            sys.exit(3)
        elif data_migrations.has_pending_migrations(db_api.get_engine()):
            print(_('Your database is not up to date. Your next step is to run `glance-manage db migrate`.'))
            sys.exit(4)
        elif not contract_heads or contract_heads not in curr_heads:
            print(_('Your database is not up to date. Your next step is to run `glance-manage db contract`.'))
            sys.exit(5)

    @args('--version', metavar='<version>', help='Database version')
    def upgrade(self, version='heads'):
        """Upgrade the database's migration level"""
        self._sync(version)

    @args('--version', metavar='<version>', help='Database version')
    def version_control(self, version=db_migration.ALEMBIC_INIT_VERSION):
        """Place a database under migration control"""
        if version is None:
            version = db_migration.ALEMBIC_INIT_VERSION
        a_config = alembic_migrations.get_alembic_config()
        alembic_command.stamp(a_config, version)
        print(_('Placed database under migration control at revision:'), version)

    @args('--version', metavar='<version>', help='Database version')
    def sync(self, version=None):
        """Perform a complete (offline) database migration"""
        global USE_TRIGGERS
        USE_TRIGGERS = False
        curr_heads = alembic_migrations.get_current_alembic_heads()
        contract = alembic_migrations.get_alembic_branch_head(db_migration.CONTRACT_BRANCH)
        if contract in curr_heads:
            print(_('Database is up to date. No migrations needed.'))
            sys.exit()
        try:
            self.expand(online_migration=False)
            self.migrate(online_migration=False)
            self.contract(online_migration=False)
            print(_('Database is synced successfully.'))
        except exception.GlanceException as e:
            sys.exit(_('Failed to sync database: ERROR: %s') % e)

    def _sync(self, version):
        """
        Place an existing database under migration control and upgrade it.
        """
        a_config = alembic_migrations.get_alembic_config()
        alembic_command.upgrade(a_config, version)
        heads = alembic_migrations.get_current_alembic_heads()
        if heads is None:
            raise exception.GlanceException('Database sync failed')
        revs = ', '.join(heads)
        if version == 'heads':
            print(_('Upgraded database, current revision(s):'), revs)
        else:
            print(_('Upgraded database to: %(v)s, current revision(s): %(r)s') % {'v': version, 'r': revs})

    def _validate_engine(self, engine):
        """Check engine is valid or not.

        MySql is only supported for online upgrade.
        Adding sqlite as engine to support existing functional test cases.

        :param engine: database engine name
        """
        if engine.engine.name not in ['mysql', 'sqlite']:
            sys.exit(_('Rolling upgrades are currently supported only for MySQL and Sqlite'))

    def expand(self, online_migration=True):
        """Run the expansion phase of a database migration."""
        if online_migration:
            self._validate_engine(db_api.get_engine())
        curr_heads = alembic_migrations.get_current_alembic_heads()
        expand_head = alembic_migrations.get_alembic_branch_head(db_migration.EXPAND_BRANCH)
        contract_head = alembic_migrations.get_alembic_branch_head(db_migration.CONTRACT_BRANCH)
        if not expand_head:
            sys.exit(_("Database expansion failed. Couldn't find head revision of expand branch."))
        elif contract_head in curr_heads:
            print(_('Database is up to date. No migrations needed.'))
            sys.exit()
        if expand_head not in curr_heads:
            self._sync(version=expand_head)
            curr_heads = alembic_migrations.get_current_alembic_heads()
            if expand_head not in curr_heads:
                sys.exit(_('Database expansion failed. Database expansion should have brought the database version up to "%(e_rev)s" revision. But, current revisions are: %(curr_revs)s ') % {'e_rev': expand_head, 'curr_revs': curr_heads})
        else:
            print(_('Database expansion is up to date. No expansion needed.'))

    def contract(self, online_migration=True):
        """Run the contraction phase of a database migration."""
        if online_migration:
            self._validate_engine(db_api.get_engine())
        curr_heads = alembic_migrations.get_current_alembic_heads()
        contract_head = alembic_migrations.get_alembic_branch_head(db_migration.CONTRACT_BRANCH)
        if not contract_head:
            sys.exit(_("Database contraction failed. Couldn't find head revision of contract branch."))
        elif contract_head in curr_heads:
            print(_('Database is up to date. No migrations needed.'))
            sys.exit()
        expand_head = alembic_migrations.get_alembic_branch_head(db_migration.EXPAND_BRANCH)
        if expand_head not in curr_heads:
            sys.exit(_('Database contraction did not run. Database contraction cannot be run before database expansion. Run database expansion first using "glance-manage db expand"'))
        if data_migrations.has_pending_migrations(db_api.get_engine()):
            sys.exit(_('Database contraction did not run. Database contraction cannot be run before data migration is complete. Run data migration using "glance-manage db migrate".'))
        self._sync(version=contract_head)
        curr_heads = alembic_migrations.get_current_alembic_heads()
        if contract_head not in curr_heads:
            sys.exit(_('Database contraction failed. Database contraction should have brought the database version up to "%(e_rev)s" revision. But, current revisions are: %(curr_revs)s ') % {'e_rev': expand_head, 'curr_revs': curr_heads})

    def migrate(self, online_migration=True):
        """Run the data migration phase of a database migration."""
        if online_migration:
            self._validate_engine(db_api.get_engine())
        curr_heads = alembic_migrations.get_current_alembic_heads()
        contract_head = alembic_migrations.get_alembic_branch_head(db_migration.CONTRACT_BRANCH)
        if contract_head in curr_heads:
            print(_('Database is up to date. No migrations needed.'))
            sys.exit()
        expand_head = alembic_migrations.get_alembic_branch_head(db_migration.EXPAND_BRANCH)
        if expand_head not in curr_heads:
            sys.exit(_('Data migration did not run. Data migration cannot be run before database expansion. Run database expansion first using "glance-manage db expand"'))
        if data_migrations.has_pending_migrations(db_api.get_engine()):
            rows_migrated = data_migrations.migrate(db_api.get_engine())
            print(_('Migrated %s rows') % rows_migrated)
        else:
            print(_('Database migration is up to date. No migration needed.'))

    @args('--path', metavar='<path>', help='Path to the directory or file where json metadata is stored')
    @args('--merge', action='store_true', help='Merge files with data that is in the database. By default it prefers existing data over new. This logic can be changed by combining --merge option with one of these two options: --prefer_new or --overwrite.')
    @args('--prefer_new', action='store_true', help='Prefer new metadata over existing. Existing metadata might be overwritten. Needs to be combined with --merge option.')
    @args('--overwrite', action='store_true', help='Drop and rewrite metadata. Needs to be combined with --merge option')
    def load_metadefs(self, path=None, merge=False, prefer_new=False, overwrite=False):
        """Load metadefinition json files to database"""
        metadata.db_load_metadefs(db_api.get_engine(), path, merge, prefer_new, overwrite)

    def unload_metadefs(self):
        """Unload metadefinitions from database"""
        metadata.db_unload_metadefs(db_api.get_engine())

    @args('--path', metavar='<path>', help='Path to the directory where json metadata files should be saved.')
    def export_metadefs(self, path=None):
        """Export metadefinitions data from database to files"""
        metadata.db_export_metadefs(db_api.get_engine(), path)

    def _purge(self, age_in_days, max_rows, purge_images_only=False):
        try:
            age_in_days = int(age_in_days)
        except ValueError:
            sys.exit(_('Invalid int value for age_in_days: %(age_in_days)s') % {'age_in_days': age_in_days})
        try:
            max_rows = int(max_rows)
        except ValueError:
            sys.exit(_('Invalid int value for max_rows: %(max_rows)s') % {'max_rows': max_rows})
        if age_in_days < 0:
            sys.exit(_('Must supply a non-negative value for age.'))
        if age_in_days >= int(time.time()) / 86400:
            sys.exit(_('Maximal age is count of days since epoch.'))
        if max_rows < -1:
            sys.exit(_('Minimal rows limit is -1.'))
        ctx = context.get_admin_context(show_deleted=True)
        try:
            if purge_images_only:
                db_api.purge_deleted_rows_from_images(ctx, age_in_days, max_rows)
            else:
                db_api.purge_deleted_rows(ctx, age_in_days, max_rows)
        except exception.Invalid as exc:
            sys.exit(exc.msg)
        except db_exc.DBReferenceError:
            sys.exit(_('Purge command failed, check glance-manage logs for more details.'))

    @args('--age_in_days', type=int, help='Purge deleted rows older than age in days')
    @args('--max_rows', type=int, help='Limit number of records to delete. All deleted rows will be purged if equals -1.')
    def purge(self, age_in_days=30, max_rows=100):
        """Purge deleted rows older than a given age from glance tables."""
        self._purge(age_in_days, max_rows)

    @args('--age_in_days', type=int, help='Purge deleted rows older than age in days')
    @args('--max_rows', type=int, help='Limit number of records to delete. All deleted rows will be purged if equals -1.')
    def purge_images_table(self, age_in_days=180, max_rows=100):
        """Purge deleted rows older than a given age from images table."""
        self._purge(age_in_days, max_rows, purge_images_only=True)