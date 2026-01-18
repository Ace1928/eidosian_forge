import sys
from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.recorder import MigrationRecorder

        Show all known migrations (or only those of the specified app_names)
        in the order they will be applied.
        