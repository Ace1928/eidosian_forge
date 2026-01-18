import os
import time
import warnings
from asgiref.local import Local
from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.db import connections, router
from django.db.utils import ConnectionRouter
from django.dispatch import Signal, receiver
from django.utils import timezone
from django.utils.formats import FORMAT_SETTINGS, reset_format_cache
from django.utils.functional import empty
from django.utils.module_loading import import_string
@receiver(setting_changed)
def update_installed_apps(*, setting, **kwargs):
    if setting == 'INSTALLED_APPS':
        from django.contrib.staticfiles.finders import get_finder
        get_finder.cache_clear()
        from django.core.management import get_commands
        get_commands.cache_clear()
        from django.template.utils import get_app_template_dirs
        get_app_template_dirs.cache_clear()
        from django.utils.translation import trans_real
        trans_real._translations = {}