import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def unset_installed_apps(self):
    """Cancel a previous call to set_installed_apps()."""
    self.app_configs = self.stored_app_configs.pop()
    self.apps_ready = self.models_ready = self.ready = True
    self.clear_cache()