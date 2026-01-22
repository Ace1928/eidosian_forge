import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from django.apps import AppConfig
from django.apps.registry import Apps
from django.apps.registry import apps as global_apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.migrations.utils import field_is_referenced, get_references
from django.db.models import NOT_PROVIDED
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
from django.db.models.options import DEFAULT_NAMES, normalize_together
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version
from .exceptions import InvalidBasesError
from .utils import resolve_relation
def _find_concrete_model_from_proxy(self, proxy_models, model_state):
    for base in model_state.bases:
        if not (isinstance(base, str) or issubclass(base, models.Model)):
            continue
        base_key = make_model_tuple(base)
        base_state = proxy_models.get(base_key)
        if not base_state:
            return base_key
        return self._find_concrete_model_from_proxy(proxy_models, base_state)