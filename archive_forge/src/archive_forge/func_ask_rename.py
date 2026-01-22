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
def ask_rename(self, model_name, old_name, new_name, field_instance):
    """Was this field really renamed?"""
    msg = 'Was %s.%s renamed to %s.%s (a %s)? [y/N]'
    return self._boolean_input(msg % (model_name, old_name, model_name, new_name, field_instance.__class__.__name__), False)