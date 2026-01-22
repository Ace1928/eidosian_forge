import pkgutil
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.connection import ConnectionDoesNotExist  # NOQA: F401
from django.utils.connection import BaseConnectionHandler
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def allow_migrate(self, db, app_label, **hints):
    for router in self.routers:
        try:
            method = router.allow_migrate
        except AttributeError:
            continue
        allow = method(db, app_label, **hints)
        if allow is not None:
            return allow
    return True