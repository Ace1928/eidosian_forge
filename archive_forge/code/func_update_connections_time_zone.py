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
def update_connections_time_zone(*, setting, **kwargs):
    if setting == 'TIME_ZONE':
        if hasattr(time, 'tzset'):
            if kwargs['value']:
                os.environ['TZ'] = kwargs['value']
            else:
                os.environ.pop('TZ', None)
            time.tzset()
        timezone.get_default_timezone.cache_clear()
    if setting in {'TIME_ZONE', 'USE_TZ'}:
        for conn in connections.all(initialized_only=True):
            try:
                del conn.timezone
            except AttributeError:
                pass
            try:
                del conn.timezone_name
            except AttributeError:
                pass
            conn.ensure_timezone()