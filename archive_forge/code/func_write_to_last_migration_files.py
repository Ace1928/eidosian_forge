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
def write_to_last_migration_files(self, changes):
    loader = MigrationLoader(connections[DEFAULT_DB_ALIAS])
    new_changes = {}
    update_previous_migration_paths = {}
    for app_label, app_migrations in changes.items():
        leaf_migration_nodes = loader.graph.leaf_nodes(app=app_label)
        if len(leaf_migration_nodes) == 0:
            raise CommandError(f'App {app_label} has no migration, cannot update last migration.')
        leaf_migration_node = leaf_migration_nodes[0]
        leaf_migration = loader.graph.nodes[leaf_migration_node]
        if leaf_migration.replaces:
            raise CommandError(f"Cannot update squash migration '{leaf_migration}'.")
        if leaf_migration_node in loader.applied_migrations:
            raise CommandError(f"Cannot update applied migration '{leaf_migration}'.")
        depending_migrations = [migration for migration in loader.disk_migrations.values() if leaf_migration_node in migration.dependencies]
        if depending_migrations:
            formatted_migrations = ', '.join([f"'{migration}'" for migration in depending_migrations])
            raise CommandError(f"Cannot update migration '{leaf_migration}' that migrations {formatted_migrations} depend on.")
        for migration in app_migrations:
            leaf_migration.operations.extend(migration.operations)
            for dependency in migration.dependencies:
                if isinstance(dependency, SwappableTuple):
                    if settings.AUTH_USER_MODEL == dependency.setting:
                        leaf_migration.dependencies.append(('__setting__', 'AUTH_USER_MODEL'))
                    else:
                        leaf_migration.dependencies.append(dependency)
                elif dependency[0] != migration.app_label:
                    leaf_migration.dependencies.append(dependency)
        optimizer = MigrationOptimizer()
        leaf_migration.operations = optimizer.optimize(leaf_migration.operations, app_label)
        previous_migration_path = MigrationWriter(leaf_migration).path
        name_fragment = self.migration_name or leaf_migration.suggest_name()
        suggested_name = leaf_migration.name[:4] + f'_{name_fragment}'
        if leaf_migration.name == suggested_name:
            new_name = leaf_migration.name + '_updated'
        else:
            new_name = suggested_name
        leaf_migration.name = new_name
        new_changes[app_label] = [leaf_migration]
        update_previous_migration_paths[app_label] = previous_migration_path
    self.write_migration_files(new_changes, update_previous_migration_paths)