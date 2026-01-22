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
def ask_initial(self, app_label):
    """Should we create an initial migration for the app?"""
    if app_label in self.specified_apps:
        return True
    try:
        app_config = apps.get_app_config(app_label)
    except LookupError:
        return self.defaults.get('ask_initial', False)
    migrations_import_path, _ = MigrationLoader.migrations_module(app_config.label)
    if migrations_import_path is None:
        return self.defaults.get('ask_initial', False)
    try:
        migrations_module = importlib.import_module(migrations_import_path)
    except ImportError:
        return self.defaults.get('ask_initial', False)
    else:
        if getattr(migrations_module, '__file__', None):
            filenames = os.listdir(os.path.dirname(migrations_module.__file__))
        elif hasattr(migrations_module, '__path__'):
            if len(migrations_module.__path__) > 1:
                return False
            filenames = os.listdir(list(migrations_module.__path__)[0])
        return not any((x.endswith('.py') for x in filenames if x != '__init__.py'))