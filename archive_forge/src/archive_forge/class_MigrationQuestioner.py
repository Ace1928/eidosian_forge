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
class MigrationQuestioner:
    """
    Give the autodetector responses to questions it might have.
    This base class has a built-in noninteractive mode, but the
    interactive subclass is what the command-line arguments will use.
    """

    def __init__(self, defaults=None, specified_apps=None, dry_run=None):
        self.defaults = defaults or {}
        self.specified_apps = specified_apps or set()
        self.dry_run = dry_run

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

    def ask_not_null_addition(self, field_name, model_name):
        """Adding a NOT NULL field to a model."""
        return None

    def ask_not_null_alteration(self, field_name, model_name):
        """Changing a NULL field to NOT NULL."""
        return None

    def ask_rename(self, model_name, old_name, new_name, field_instance):
        """Was this field really renamed?"""
        return self.defaults.get('ask_rename', False)

    def ask_rename_model(self, old_model_state, new_model_state):
        """Was this model really renamed?"""
        return self.defaults.get('ask_rename_model', False)

    def ask_merge(self, app_label):
        """Should these migrations really be merged?"""
        return self.defaults.get('ask_merge', False)

    def ask_auto_now_add_addition(self, field_name, model_name):
        """Adding an auto_now_add field to a model."""
        return None

    def ask_unique_callable_default_addition(self, field_name, model_name):
        """Adding a unique field with a callable default."""
        return None