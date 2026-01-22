import datetime
import importlib
import os
import sys
from django.apps import apps
from django.core.management.base import OutputWrapper
from django.db.models import NOT_PROVIDED
from django.utils import timezone
from django.utils.version import get_docs_version
from .loader import MigrationLoader
def ask_not_null_addition(self, field_name, model_name):
    self.log_lack_of_migration(field_name, model_name, 'it is impossible to add a non-nullable field without specifying a default')
    sys.exit(3)