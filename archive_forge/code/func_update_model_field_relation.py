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
def update_model_field_relation(self, model, model_key, field_name, field, concretes):
    remote_model_key = resolve_relation(model, *model_key)
    if remote_model_key[0] not in self.real_apps and remote_model_key in concretes:
        remote_model_key = concretes[remote_model_key]
    relations_to_remote_model = self._relations[remote_model_key]
    if field_name in self.models[model_key].fields:
        assert field_name not in relations_to_remote_model[model_key]
        relations_to_remote_model[model_key][field_name] = field
    else:
        del relations_to_remote_model[model_key][field_name]
        if not relations_to_remote_model[model_key]:
            del relations_to_remote_model[model_key]