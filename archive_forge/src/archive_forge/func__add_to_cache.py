from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def _add_to_cache(self, using, ct):
    """Insert a ContentType into the cache."""
    key = (ct.app_label, ct.model)
    self._cache.setdefault(using, {})[key] = ct
    self._cache.setdefault(using, {})[ct.id] = ct