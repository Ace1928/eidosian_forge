import os
import sys
import warnings
from itertools import takewhile
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.utils import run_formatters
from django.db import DEFAULT_DB_ALIAS, OperationalError, connections, router
from django.db.migrations import Migration
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.migration import SwappableTuple
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import (
from django.db.migrations.state import ProjectState
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.migrations.writer import MigrationWriter
def write_migration_files(self, changes, update_previous_migration_paths=None):
    """
        Take a changes dict and write them out as migration files.
        """
    directory_created = {}
    for app_label, app_migrations in changes.items():
        if self.verbosity >= 1:
            self.log(self.style.MIGRATE_HEADING("Migrations for '%s':" % app_label))
        for migration in app_migrations:
            writer = MigrationWriter(migration, self.include_header)
            if self.verbosity >= 1:
                migration_string = self.get_relative_path(writer.path)
                self.log('  %s\n' % self.style.MIGRATE_LABEL(migration_string))
                for operation in migration.operations:
                    self.log('    - %s' % operation.describe())
                if self.scriptable:
                    self.stdout.write(migration_string)
            if not self.dry_run:
                migrations_directory = os.path.dirname(writer.path)
                if not directory_created.get(app_label):
                    os.makedirs(migrations_directory, exist_ok=True)
                    init_path = os.path.join(migrations_directory, '__init__.py')
                    if not os.path.isfile(init_path):
                        open(init_path, 'w').close()
                    directory_created[app_label] = True
                migration_string = writer.as_string()
                with open(writer.path, 'w', encoding='utf-8') as fh:
                    fh.write(migration_string)
                    self.written_files.append(writer.path)
                if update_previous_migration_paths:
                    prev_path = update_previous_migration_paths[app_label]
                    rel_prev_path = self.get_relative_path(prev_path)
                    if writer.needs_manual_porting:
                        migration_path = self.get_relative_path(writer.path)
                        self.log(self.style.WARNING(f'Updated migration {migration_path} requires manual porting.\nPrevious migration {rel_prev_path} was kept and must be deleted after porting functions manually.'))
                    else:
                        os.remove(prev_path)
                        self.log(f'Deleted {rel_prev_path}')
            elif self.verbosity == 3:
                self.log(self.style.MIGRATE_HEADING("Full migrations file '%s':" % writer.filename))
                self.log(writer.as_string())
    run_formatters(self.written_files)