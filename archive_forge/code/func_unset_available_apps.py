import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def unset_available_apps(self):
    """Cancel a previous call to set_available_apps()."""
    self.app_configs = self.stored_app_configs.pop()
    self.clear_cache()