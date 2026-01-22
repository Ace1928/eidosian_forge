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
def ask_merge(self, app_label):
    return self._boolean_input('\nMerging will only work if the operations printed above do not conflict\n' + 'with each other (working on different fields or models)\n' + 'Should these migration branches be merged? [y/N]', False)