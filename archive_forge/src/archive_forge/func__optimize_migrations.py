import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def _optimize_migrations(self):
    for app_label, migrations in self.migrations.items():
        for m1, m2 in zip(migrations, migrations[1:]):
            m2.dependencies.append((app_label, m1.name))
    for migrations in self.migrations.values():
        for migration in migrations:
            migration.dependencies = list(set(migration.dependencies))
    for app_label, migrations in self.migrations.items():
        for migration in migrations:
            migration.operations = MigrationOptimizer().optimize(migration.operations, app_label)